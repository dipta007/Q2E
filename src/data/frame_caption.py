from datasets import load_from_disk

from .constants import ARGS
from .utils import save_data
from .data_utils.video_planner import get_all_frame_captions as get_captions


def get_frame_captions(video_ids):
    print("Getting captions...")
    captions = get_captions(video_ids)
    print("Captions processed")
    return captions


def main():
    ds = load_from_disk(ARGS.dataset_dir)

    video_ids = set()
    for i in range(len(ds)):
        video_id = ds[i]["metadata"]["video_id"]
        video_ids.add(video_id)

    video_ids = list(video_ids)
    frame_captions = get_frame_captions(video_ids)

    num_of_frames_col = []
    frame_captions_col = []
    video_ids = []
    for i in range(len(ds)):
        video_id = ds[i]["metadata"]["video_id"]
        frame_caption = frame_captions[video_id]

        num_of_frames_col.append(frame_caption["num_of_frames"])
        frame_captions_col.append(frame_caption["frame_captions"])
        video_ids.append(video_id)

    ds = ds.add_column("num_of_frames", num_of_frames_col)
    ds = ds.add_column("frame_captions", frame_captions_col)
    ds = ds.add_column("video_id", video_ids)

    print(ds)
    save_data(ds.to_dict())


if __name__ == "__main__":
    main()
