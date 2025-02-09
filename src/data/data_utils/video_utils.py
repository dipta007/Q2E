import os

import pandas as pd

from ..constants import ARGS


def download(df):
    video_ids = []
    filtered_df = []
    for i, row in df.iterrows():
        video_id = row["video_id"]
        if os.path.isfile(f"{ARGS.video_dir}/{video_id}.mp4"):
            video_ids.append(video_id)
            row["video_id"] = video_id
            filtered_df.append(row)

    print(f"Downloaded {len(video_ids)}/{len(df)} videos")
    return pd.DataFrame(filtered_df)


if __name__ == "__main__":
    pass
