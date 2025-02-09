from collections import defaultdict

import pandas as pd

from .constants import ARGS
from .data_utils.text_planner import get_prequel_sequel_during
from .data_utils.video_utils import download
from .utils import save_data

metadatas = []


def get_metadata(df):
    global metadatas
    metadatas = [{} for _ in range(len(df))]
    for i, row in df.iterrows():
        columns = df.columns.tolist()
        metadata = {}
        for col in columns:
            if col == "Unnamed: 0":
                continue
            metadata[col] = row[col]
        metadatas[i] = metadata


def get_text_planner_data(df, data):
    global metadatas
    queries = df[ARGS.query_col].tolist()
    queries = list(set(queries))

    prequel, sequel, during = get_prequel_sequel_during(queries)

    for i, row in df.iterrows():
        try:
            query = row[ARGS.query_col]

            data["query"].append(query)
            data["prequel"].append(prequel[query]["processed_text"])
            data["sequel"].append(sequel[query]["processed_text"])
            data["during"].append(during[query]["processed_text"])

            metadatas[i]["prequel_prompt"] = prequel[query]["prompt"]
            metadatas[i]["sequel_prompt"] = sequel[query]["prompt"]
            metadatas[i]["during_prompt"] = during[query]["prompt"]

            metadatas[i]["prequel_raw_generation"] = prequel[query]["generated_text"]
            metadatas[i]["sequel_raw_generation"] = sequel[query]["generated_text"]
            metadatas[i]["during_raw_generation"] = during[query]["generated_text"]
        except Exception as e:
            print(e)
            print(f"Error for query: {query}")

    return data


def main():
    data = defaultdict(list)

    dataframe = "dataset"
    if ARGS.split == "dev":
        dataframe = "dev_dataset"

    df = pd.read_csv(f"{ARGS.data_dir}/{dataframe}.csv")
    print("Non-filtered df length: ", len(df))
    df = download(df)
    print("Filtered df length: ", len(df))
    df.reset_index(drop=True, inplace=True)
    df.to_csv(f"{ARGS.dataset_dir}_raw.csv", index=False)

    get_metadata(df)

    print("Query Planner data Processing...")
    data = get_text_planner_data(df, data)
    save_data(data)
    print("Query Planner data processed")

    global metadatas
    data["metadata"] = metadatas

    print("Query data processed")
    save_data(data)
    print("Data saved")


if __name__ == "__main__":
    main()
