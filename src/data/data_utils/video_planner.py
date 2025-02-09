import gc
import os
from pathlib import Path

import cv2
import torch
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from scenedetect import AdaptiveDetector, detect
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..constants import ARGS
from ..utils import get_asr_refined, get_asr_summary

model, processor = None, None


def init_model():
    print("Initializing InternVL VLM")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    model = LLM(
        model=ARGS.gen_vlm_id,
        max_model_len=ARGS.gen_max_model_len,
        dtype=dtype,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    processor = AutoTokenizer.from_pretrained(ARGS.gen_vlm_id, trust_remote_code=True)
    print("InternVL VLM Initialized")
    return model, processor


def destroy_vlm():
    global model, processor
    del model
    del processor
    model, processor = None, None
    torch.cuda.empty_cache()
    gc.collect()


def get_prompt(prev_captions, video_id, template_name):
    if len(prev_captions) == 0:
        template_name = template_name.replace("contextualized_", "")

    env = Environment(loader=FileSystemLoader("src/data/prompts"))
    template = env.get_template(f"{template_name}.jinja")

    prev_caption = prev_captions[-1] if prev_captions else ""
    ASR_summary = get_asr_summary(video_id)
    ASR_refined = get_asr_refined(video_id)
    user_prompt = template.render(
        {
            "prev_caption": prev_caption,
            "ASR_summary": ASR_summary,
            "ASR_refined": ASR_refined,
        }
    )
    messages = [{"role": "user", "content": f"<image>\n{user_prompt}"}]
    global model, processor
    if not processor:
        model, processor = init_model()
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate(prompts):
    global model, processor

    if model is None:
        model, processor = init_model()

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [processor.convert_tokens_to_ids(i) for i in stop_tokens]
    sampling_params = SamplingParams(
        temperature=ARGS.gen_temperature,
        top_p=ARGS.gen_top_p,
        max_tokens=ARGS.gen_max_tokens,
        stop_token_ids=stop_token_ids,
    )

    outputs = model.generate(prompts, sampling_params=sampling_params)

    results = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text = generated_text.strip()

        results.append(generated_text)

        if ARGS.debug:
            print("=" * 80)
            print("---- Prompt")
            print(prompt)
            print("---- Generated Text")
            print(generated_text)
            print("=" * 80)
            print()

    return results


def uniformly_sample_frames(video_id, num_clips):
    try:
        if os.path.exists(f"{ARGS.data_dir}/frames/{num_clips}/{video_id}"):
            return [Image.open(f"{ARGS.data_dir}/frames/{num_clips}/{video_id}/frame_{i}.jpg") for i in range(num_clips)]
    except:
        pass
    video_path = f"{ARGS.video_dir}/{video_id}.mp4"
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // num_clips)

    images = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # discard the first frame
        if frame_count and frame_count % frame_step == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            images.append(image)

        frame_count += 1

    images = images + [images[-1]] * (num_clips - len(images))

    cap.release()

    frame_path = f"{ARGS.data_dir}/frames/{num_clips}/{video_id}"
    Path(frame_path).mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        image.save(f"{frame_path}/frame_{i}.jpg")

    return images


def scene_detect_sample_frames(video_id):
    video_path = f"{ARGS.video_dir}/{video_id}.mp4"
    try:
        if os.path.exists(f"{ARGS.data_dir}/SD_frames/{video_id}"):
            return [
                Image.open(f"{ARGS.data_dir}/SD_frames/{video_id}/frame_{i}.jpg")
                for i in range(len(os.listdir(f"{ARGS.data_dir}/SD_frames/{video_id}")))
            ]
    except:
        pass

    def extract_frames(video_path, frame_nums):
        images = []
        cap = cv2.VideoCapture(video_path)

        for frame_num in frame_nums:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                raise Exception(f"Failed to read frame {frame_num}")

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            images.append(image)

        cap.release()
        return images

    scene_list = detect(video_path, AdaptiveDetector())

    frame_nums = []
    for i, scene in enumerate(scene_list):
        frame_start, frame_end = scene[0].get_frames(), scene[1].get_frames()
        middle_frame = (frame_start + frame_end) // 2
        frame_nums.append(middle_frame)

    images = extract_frames(video_path, frame_nums)

    frame_path = f"{ARGS.data_dir}/SD_frames/{video_id}"
    Path(frame_path).mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        image.save(f"{frame_path}/frame_{i}.jpg")

    return images


def get_all_frame_captions(video_ids):
    template_name = "contextualized_frame_caption"
    if ARGS.with_asr:
        template_name = "contextualized_frame_caption+ASR"

    result = {}

    for video_id in video_ids:
        result[video_id] = {"frame_captions": [], "num_of_frames": 0}

    videoID_to_frames = {}
    curr_video_ids = []
    for video_id in tqdm(video_ids, desc="Extracting frames"):
        if ARGS.frame_selection_method == "uniform":
            frames = uniformly_sample_frames(video_id, ARGS.num_of_frames)
        elif ARGS.frame_selection_method == "scene_detect":
            frames = scene_detect_sample_frames(video_id)
            if len(frames) == 0:
                frames = uniformly_sample_frames(video_id, ARGS.num_of_frames)
        else:
            raise NotImplementedError(f"Frame selection method {ARGS.frame_selection_method} not implemented")

        videoID_to_frames[video_id] = frames

        curr_video_ids.append(video_id)

        if len(curr_video_ids) == 512 or video_id == video_ids[-1]:
            mx_frames = max([len(videoID_to_frames[video_id]) for video_id in curr_video_ids])
            for video_id in curr_video_ids:
                result[video_id]["num_of_frames"] = len(videoID_to_frames[video_id])

            for i in tqdm(
                range(mx_frames),
                desc=f"Generating clip captions for {len(curr_video_ids)} videos",
            ):
                curr_prompts = []
                for video_id in curr_video_ids:
                    if i >= len(videoID_to_frames[video_id]):  # in case of scene detect, num_of_frames can be variable
                        continue
                    frame = videoID_to_frames[video_id][i]
                    prev_captions = result[video_id]["frame_captions"][-1:]
                    prompt = get_prompt(prev_captions, video_id, template_name)
                    curr_prompts.append({"prompt": prompt, "multi_modal_data": {"image": frame}})

                print(f"============ Prompt Size Frame {i} ============")
                mx = max([len(p["prompt"].split()) for p in curr_prompts])
                print(mx)
                print("=====================================")
                curr_captions = generate(curr_prompts)

                j = 0
                for video_id in curr_video_ids:
                    if i >= len(videoID_to_frames[video_id]):
                        continue
                    result[video_id]["frame_captions"].append(curr_captions[j])
                    j += 1

            curr_video_ids = []
            videoID_to_frames = {}

    for video_id in result.keys():
        result[video_id]["frame_captions"] = result[video_id]["frame_captions"][: result[video_id]["num_of_frames"]]
    print("============ Mean Num of Clips ============")
    print(sum([result[video_id]["num_of_frames"] for video_id in result.keys()]) / len(result.keys()))
    print("=====================================")

    return result


if __name__ == "__main__":
    pass
