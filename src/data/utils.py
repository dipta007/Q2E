import os
from datasets import Dataset
from .constants import ARGS


def save_data(data):
    out_data = Dataset.from_dict(data)
    out_data.save_to_disk(ARGS.dataset_dir)
    print("====================================")
    print(f"Data saved to {ARGS.dataset_dir}")
    print(out_data)
    print("====================================")
    out_data_df = out_data.to_pandas()
    out_data_df.to_csv(f"{ARGS.dataset_dir}.csv", index=False)


def get_asr_summary(video_id):
    transcript_dir = f"{ARGS.data_dir}/transcripts/summarized"
    transcript_file = f"{transcript_dir}/{video_id}.txt"
    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()
    except:
        transcript = "Not Generated"
    return transcript


def get_asr_refined(video_id):
    transcript_dir = f"{ARGS.data_dir}/transcripts"
    transcript_file = f"{transcript_dir}/{video_id}.txt"
    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()
    except:
        transcript = "Not Generated"
    return transcript


def get_all_asr(video_id):
    transcript_dir = f"{ARGS.data_dir}/transcripts"
    asr = {}
    dirs = os.listdir(transcript_dir)
    for dir in dirs:
        transcript_file = f"{transcript_dir}/{dir}/{video_id}.txt"
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript = f.read()
        except:
            transcript = "Not Generated"
        asr[dir] = transcript
    return asr
