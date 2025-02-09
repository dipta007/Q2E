from datasets import load_from_disk

from .constants import ARGS

from .data_utils.text_planner import generate
from .query_decomp import save_data
from .utils import get_all_asr, get_asr_refined, get_asr_summary


def main():
    ds = load_from_disk(ARGS.dataset_dir)

    queries = []
    asrs = []
    for i in range(len(ds)):
        frame_captions = ds[i]["frame_captions"]
        video_id = ds[i]["video_id"]
        ASR_summary = get_asr_summary(video_id)
        ASR_refined = get_asr_refined(video_id)

        query = "# Frame Descriptions:\n\n"
        for i, frame_caption in enumerate(frame_captions):
            query += f"## Frame {i + 1} Description:\n"
            query += f"{frame_caption}\n\n"

        queries.append({"query": query, "ASR_summary": ASR_summary, "ASR_refined": ASR_refined})

        if ARGS.with_asr:
            asr = get_all_asr(video_id)
            asrs.append(asr)

    template_name = "summary_video_caption"
    if ARGS.with_asr:
        template_name = "summary_video_caption+ASR"
    outputs = generate(template_name, queries)

    ds = ds.add_column("frame2video_caption", outputs["generated_text"])
    if ARGS.with_asr:
        ds = ds.add_column("asr", asrs)

    save_data(ds.to_dict())


if __name__ == "__main__":
    main()
