import os

os.environ["WANDB_MODE"] = "offline"

data_dirs = [
    "data/MSR-VTT-1kA",
    "data/MultiVENT",
]

gen_llm_ids = [
    # "meta-llama/Llama-3.3-70B-Instruct",
    # "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    # "Qwen/Qwen2.5-32B-Instruct",
    # "Qwen/Qwen2.5-72B-Instruct"
]

gen_vlm_ids = [
    # "OpenGVLab/InternVL2_5-38B",
    # "OpenGVLab/InternVL2_5-26B",
    # "OpenGVLab/InternVL2_5-8B",
    # "OpenGVLab/InternVL2_5-4B",
    # "OpenGVLab/InternVL2_5-2B",
    "OpenGVLab/InternVL2_5-1B",
]

frame_selection_methods = [
    "uniform",
    # 'scene_detect'
]

opts_num_of_frames = [
    # 2,
    4,
    # 8,
    # 16,
    # 32,
    # 64,
]

with_asrs = [
    False,
    True,
]


def run(cmd, args, desc):
    cmd = f"{cmd} {args}"
    print(f"Processing {desc}")
    print("====")
    print(cmd)
    print("====")
    ret = os.system(cmd)
    if not ret == 0:
        print(f"$$$$ Error in processing {desc}: {ret} $$$$")
    return ret


def pipeline(data_dir, gen_llm_id, gen_vlm_id, frame_selection_method, num_of_frames, with_asr, debug=False):
    video_dir = f"{data_dir}/videos"

    cmd0 = f"""python -m src.data.transcribe_audios \
        --video_dir={video_dir}"""

    args = f"""--data_dir="{data_dir}" \
        --video_dir="{video_dir}" \
        --gen_llm_id="{gen_llm_id}" \
        --gen_vlm_id="{gen_vlm_id}" \
        --frame_selection_method="{frame_selection_method}" \
        --num_of_frames="{num_of_frames}" """
    if with_asr:
        args += " --with_asr"
    if debug:
        args += " --debug"

    cmd1 = "python -m src.data.query_decomp --gen_max_model_len=2048"
    cmd2 = "python -m src.data.frame_caption --gen_max_model_len=16384"
    cmd3 = "python -m src.data.frame2video_caption --gen_max_model_len=16384"

    print(
        f"#### Running pipeline for data_dir={data_dir}, gen_llm_id={gen_llm_id}, gen_vlm_id={gen_vlm_id}, frame_selection_method={frame_selection_method}, with_asr={with_asr} ####"
    )

    if with_asr and run(cmd0, "", "transcribing videos") != 0:
        return

    if run(cmd1, args, "processing raw data") != 0:
        return

    if run(cmd2, args, "captioning frames") != 0:
        return

    if run(cmd3, args, "captioning videos") != 0:
        return

    print("Successfully Done")
    print("\n\n\n")


for data_dir in data_dirs:
    for gen_llm_id in gen_llm_ids:
        for gen_vlm_id in gen_vlm_ids:
            for frame_selection_method in frame_selection_methods:
                for num_of_frames in opts_num_of_frames:
                    for with_asr in with_asrs:
                        pipeline(data_dir, gen_llm_id, gen_vlm_id, frame_selection_method, num_of_frames, with_asr=with_asr, debug=False)
