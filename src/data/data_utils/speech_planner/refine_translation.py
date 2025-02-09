import os

import torch
from jinja2 import Environment, FileSystemLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

llm, tokenizer = None, None


def init_model(args):
    print("starting LLM")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    model = LLM(
        model=args.gen_llm_id,
        dtype=dtype,
        max_model_len=args.gen_max_model_len,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.gen_llm_id)
    print("LLM started")
    return model, tokenizer


def get_prompt(data, template):
    global tokenizer
    original, whisper_translated, llm_translated = data
    user_prompt = template.render({"orig_transcript": original, "whisper_translation": whisper_translated, "llm_translation": llm_translated})

    messages = [
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt


def generate(args, prompt_template, prompts):
    global llm, tokenizer
    if llm is None:
        llm, tokenizer = init_model(args)

    if args.debug:
        print("*" * 80)
        print("Model Name:", args.gen_llm_id)
        print("Device:", torch.cuda.get_device_name())
        print("Device Count:", torch.cuda.device_count())
        print("Temperature:", args.gen_temperature)
        print("Top P:", args.gen_top_p)
        print("Max Tokens:", args.gen_max_tokens)
        print("Dtype:", args.gen_dtype)
        print("Generation format: ", prompt_template)
        print("*" * 80)

    env = Environment(loader=FileSystemLoader("src/data/prompts"))
    template = env.get_template(f"{prompt_template}.jinja")

    prompts = [get_prompt(data, template) for data in prompts]

    print("============ Prompt Size ============")
    mx = max([len(p.split()) for p in prompts])
    print(mx)
    print("=====================================")

    sampling_params = SamplingParams(
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        max_tokens=args.gen_max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text = generated_text.strip()
        generated_texts.append(generated_text)

        if args.debug:
            print("=" * 80)
            print("---- Prompt")
            print(prompt)
            print("---- Generated Text")
            print(generated_text)
            print("=" * 80)
            print()

    return generated_texts


def refine_translation(args):
    video_files = os.listdir(args.video_dir)
    prompts = []
    prompted_video_files = []
    for video_file in video_files:
        id = video_file.replace(".mp4", "")

        orig_transcript_path = os.path.join(args.orig_transcript_dir, f"{id}.txt")
        llm_translated_transcript_path = os.path.join(args.llm_translated_transcript_dir, f"{id}.txt")
        whisper_translated_transcript_path = os.path.join(args.whisper_translated_transcript_dir, f"{id}.txt")
        refined_transcript_path = os.path.join(args.refined_transcript_dir, f"{id}.txt")

        with open(orig_transcript_path, "r") as f:
            orig_transcript = f.read()
            if len(orig_transcript.strip()) == 0:
                orig_transcript = "Not Available"

        with open(llm_translated_transcript_path, "r") as f:
            llm_translated_transcript_path = f.read()
            if len(llm_translated_transcript_path.strip()) == 0:
                llm_translated_transcript_path = "Not Available"

        with open(whisper_translated_transcript_path, "r") as f:
            whisper_translated_transcript = f.read()
            if len(whisper_translated_transcript.strip()) == 0:
                whisper_translated_transcript = "Not Available"

        if not os.path.exists(refined_transcript_path):
            prompted_video_files.append(video_file)
            prompts.append((orig_transcript, whisper_translated_transcript, llm_translated_transcript_path))

    print(f"Refining {len(prompted_video_files)} transcripts")

    refines = generate(args, "refine_translation", prompts)
    na = 0
    for video_file, refine in zip(prompted_video_files, refines):
        id = video_file.replace(".mp4", "")
        refined_transcript_path = os.path.join(args.refined_transcript_dir, f"{id}.txt")

        if refine == "Not Available":
            na += 1

        with open(refined_transcript_path, "w", encoding="utf-8") as f:
            f.write(refine)

    print("Number of videos with no refine:", na)
    print("Percent of videos with no refine:", na / len(prompted_video_files) * 100)


if __name__ == "__main__":
    refine_translation()
