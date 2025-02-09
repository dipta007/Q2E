import argparse
import json
from pathlib import Path

from .data_utils.speech_planner.refine_translation import refine_translation
from .data_utils.speech_planner.transcribe import transcribe
from .data_utils.speech_planner.translate import translate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--gen_llm_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--gen_temperature", type=float, default=0.8)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--gen_max_tokens", type=int, default=2048)
    parser.add_argument("--gen_max_model_len", type=int, default=26000)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    args.base_dir = str(Path(args.video_dir).parent)
    args.audio_dir = f"{args.base_dir}/audios"
    args.transcript_dir = f"{args.base_dir}/transcripts"
    args.orig_transcript_dir = f"{args.transcript_dir}/original"
    args.whisper_translated_transcript_dir = f"{args.transcript_dir}/translated_whisper"
    args.llm_translated_transcript_dir = f"{args.transcript_dir}/translated_llm"
    args.refined_transcript_dir = f"{args.transcript_dir}/refined"

    print("==== Args ====")
    print(json.dumps(vars(args), indent=2))
    print("==============")

    Path(args.audio_dir).mkdir(parents=True, exist_ok=True)
    Path(args.orig_transcript_dir).mkdir(parents=True, exist_ok=True)
    Path(args.whisper_translated_transcript_dir).mkdir(parents=True, exist_ok=True)
    Path(args.llm_translated_transcript_dir).mkdir(parents=True, exist_ok=True)
    Path(args.refined_transcript_dir).mkdir(parents=True, exist_ok=True)

    print("====> Transcribing...")
    transcribe(args)
    print("====> Translating...")
    translate(args)
    print("====> Refining...")
    refine_translation(args)
    print("====> Done Transcribing, Translating and Refining")
