import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk

ARGS = None


def get_dataset_dir(args):
    dirs = ["Q2E"]
    if args.gen_llm_id == "meta-llama/Llama-3.3-70B-Instruct":
        dirs.append("LLAMA_3.3_70B")
    elif args.gen_llm_id == "meta-llama/Meta-Llama-3.1-70B-Instruct":
        dirs.append("LLAMA_70B")
    elif args.gen_llm_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        dirs.append("LLAMA_8B")
    elif args.gen_llm_id == "meta-llama/Llama-3.2-1B-Instruct":
        dirs.append("LLAMA_1B")
    elif args.gen_llm_id == "meta-llama/Llama-3.2-3B-Instruct":
        dirs.append("LLAMA_3B")
    elif args.gen_llm_id == "Qwen/Qwen2.5-72B-Instruct":
        dirs.append("Qwen_72B")
    elif args.gen_llm_id == "Qwen/Qwen2.5-32B-Instruct":
        dirs.append("Qwen_32B")
    elif args.gen_llm_id == "Qwen/Qwen2.5-14B-Instruct":
        dirs.append("Qwen_14B")
    elif args.gen_llm_id == "Qwen/Qwen2.5-7B-Instruct":
        dirs.append("Qwen_7B")
    elif args.gen_llm_id == "Qwen/Qwen2.5-3B-Instruct":
        dirs.append("Qwen_3B")
    elif args.gen_llm_id == "Qwen/Qwen2.5-1.5B-Instruct":
        dirs.append("Qwen_1.5B")
    elif args.gen_llm_id == "Qwen/Qwen2.5-0.5B-Instruct":
        dirs.append("Qwen_0.5B")
    else:
        raise ValueError("Invalid args.gen_llm_id")

    if args.gen_vlm_id == "OpenGVLab/InternVL2_5-78B":
        dirs.append("InternVL_78B")
    elif args.gen_vlm_id == "OpenGVLab/InternVL2_5-38B":
        dirs.append("InternVL_38B")
    elif args.gen_vlm_id == "OpenGVLab/InternVL2_5-26B":
        dirs.append("InternVL_26B")
    elif args.gen_vlm_id == "OpenGVLab/InternVL2_5-8B":
        dirs.append("InternVL_8B")
    elif args.gen_vlm_id == "OpenGVLab/InternVL2_5-4B":
        dirs.append("InternVL_4B")
    elif args.gen_vlm_id == "OpenGVLab/InternVL2_5-2B":
        dirs.append("InternVL_2B")
    elif args.gen_vlm_id == "OpenGVLab/InternVL2_5-1B":
        dirs.append("InternVL_1B")
    else:
        raise ValueError("Invalid args.gen_vlm_id")

    if args.frame_selection_method == "uniform":
        dirs.append("Funiform")
    elif args.frame_selection_method == "scene_detect":
        dirs.append("Fscene_detect")
    else:
        raise ValueError("Invalid ARGS.frame_selection_method")

    dirs.append(str(args.num_of_frames))

    if args.with_asr:
        dirs.append("ASR")
    else:
        dirs.append("noASR")

    return "_".join(dirs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)

    parser.add_argument("--gen_llm_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--gen_vlm_id", type=str, default="OpenGVLab/InternVL2_5-38B")
    parser.add_argument("--with_asr", action="store_true", default=False)

    parser.add_argument("--gen_temperature", type=float, default=0.8)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--gen_max_tokens", type=int, default=2048)
    parser.add_argument("--gen_max_model_len", type=int, default=16384)

    parser.add_argument(
        "--frame_selection_method",
        type=str,
        default="uniform",
        choices=["uniform", "scene_detect"],
    )
    parser.add_argument("--num_of_frames", type=int, default=16)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    args.query_col = "query"

    args.dataset_path = f"{args.data_dir}/{get_dataset_dir(args)}"
    args.dataset_dir = f"{args.dataset_path}/dataset"
    Path(args.dataset_dir).mkdir(parents=True, exist_ok=True)

    ds = None
    try:
        ds = load_from_disk(args.dataset_dir)
        print("=" * 25)
        print(f"Dataset: {args.dataset_dir}")
        print(ds)
        print("=" * 25)
    except Exception:
        print("Dataset does not exist. Proceeding...")
    finally:
        if ds and len(ds.column_names) == 10:
            sys.exit("Dataset already exists. Exiting...")
        elif ds and len(ds.column_names) == 9 and args.with_asr:
            sys.exit("Dataset already exists. Exiting...")

    args_dict = vars(args)
    with open(f"{args.dataset_path}/data_process_args.json", "w") as f:
        json.dump(args_dict, f, indent=2)

    print("==== Args ====")
    print(json.dumps(args_dict, indent=2))
    print("==============")

    return args


if ARGS is None:
    ARGS = get_args()
