import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk

from ..evaluation import retrieval_score
from ..fusion_score import fusion_exp_entropy, fusion_inverse_entropy, fusion_reciprocal_rank
from ..text_embedder import get_many_to_many_score
from ..utils import get_git_commit_hash
from .vision_embedder import get_query_vs_video_score

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.CRITICAL)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=str, required=True)
    parser.add_argument("--dataset_dir", "-d", type=str, required=True)
    parser.add_argument("--text_emb_type", type=str, choices=["colbert", "sentence-transformers"], default="colbert")
    parser.add_argument("--text_emb_model", type=str, default="hltcoe/plaidx-large-eng-tdist-mt5xxl-engeng")

    parser.add_argument("--aggregation_methods", type=str, default="")
    parser.add_argument("--softmax", type=str, default="pre", choices=["pre", "post", "none"])
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    if args.aggregation_methods:
        args.aggregation_methods = args.aggregation_methods.split(",")

    if args.text_emb_model.startswith("sentence-transformers"):
        args.text_emb_type = "sentence-transformers"
    else:
        args.text_emb_type = "colbert"

    args.dataset_path = os.path.join(args.dataset_dir, "dataset")

    args.git_hash = get_git_commit_hash()

    # load args from training
    with open(f"{args.dataset_dir}/data_process_args.json", "r") as f:
        args_dict = json.load(f)

    for key, value in args_dict.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


ARGS = get_args()


def get_data(ds):
    queries = []
    prequels = []
    sequels = []
    durings = []
    whole_video_captions = []
    clip_captions = []
    video_ids = []
    llm_translation_asrs = []
    whisper_translation_asrs = []
    refined_translation_asrs = []

    already_seen_query = set()
    already_seen_video = set()
    for i, row in enumerate(ds):
        query = row["query"]
        if query not in already_seen_query:
            already_seen_query.add(query)

            queries.append(query)
            prequels.append(row["prequel"])
            sequels.append(row["sequel"])
            durings.append(row["during"])

        video_id = row["video_id"]
        if video_id not in already_seen_video:
            already_seen_video.add(video_id)

            video_ids.append(video_id)
            if "whole_video_caption" in row:
                whole_video_captions.append(row["whole_video_caption"])
            else:
                whole_video_captions.append(row["clip2video_caption"])
            clip_captions.append(row["clip_captions"])

            if "asr" in row:
                llm_translation_asrs.append(row["asr"]["translated_llm"])
                whisper_translation_asrs.append(row["asr"]["translated_whisper"])
                refined_translation_asrs.append(row["asr"]["refined"])

    query_to_videoind = defaultdict(list)
    for i, row in enumerate(ds):
        query = row["query"]
        video_id = row["video_id"]
        query_to_videoind[query].append(video_ids.index(video_id))

    return (
        queries,
        prequels,
        sequels,
        durings,
        whole_video_captions,
        clip_captions,
        query_to_videoind,
        video_ids,
        llm_translation_asrs,
        whisper_translation_asrs,
        refined_translation_asrs,
    )


cache = {}


def call_func(param, **kwargs):
    print(f"Calling function: {param}", file=sys.stderr)
    if param == "query_vs_video":
        return get_query_vs_video_score(args=ARGS, queries=kwargs["queries"], video_ids=kwargs["video_ids"])
    if param == "query_vs_captions":
        return get_many_to_many_score(ARGS, kwargs["queries"], kwargs["captions"])
    elif param == "during_vs_captions":
        return get_many_to_many_score(ARGS, kwargs["durings"], kwargs["captions"])
    elif param == "prequel_vs_captions":
        return get_many_to_many_score(ARGS, kwargs["prequels"], kwargs["captions"])
    elif param == "sequel_vs_captions":
        return get_many_to_many_score(ARGS, kwargs["sequels"], kwargs["captions"])
    else:
        raise ValueError(f"Invalid param: {param}")


def get_similarity_matrix(queries, prequels, sequels, durings, captions, video_ids, params=[], aggregation="mean"):
    assert len(params) > 0, "At least one similarity score should be calculated"
    # Get similarity scores
    results = []
    pickle_data = {}
    for param in params:
        if param not in cache:
            cache[param] = call_func(
                param, queries=queries, prequels=prequels, sequels=sequels, durings=durings, captions=captions, video_ids=video_ids
            )
            if ARGS.softmax == "pre":
                # cache[param] = cache[param] * softmax(cache[param], axis=0)
                cache[param] = torch.nn.functional.softmax(cache[param], dim=0)
        results.append(cache[param])
        pickle_data[param] = cache[param]

    # Aggregate the similarity scores
    if aggregation == "mean":
        sim_matrix = torch.stack(results, dim=0).mean(dim=0)
    elif aggregation == "max":
        sim_matrix = torch.stack(results, dim=0).max(dim=0).values
    elif aggregation == "inv_entropy":
        sim_matrix = fusion_inverse_entropy(ARGS, results, params, queries, video_ids)
    elif aggregation == "exp_entropy":
        sim_matrix = fusion_exp_entropy(ARGS, results, params, queries, video_ids)
    elif aggregation == "rrf":
        sim_matrix = fusion_reciprocal_rank(ARGS, results, params, queries, video_ids)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation}")

    if ARGS.softmax == "pre":
        sim_matrix_dsl = sim_matrix  # softmax is done during embedding
    elif ARGS.softmax == "post":
        sim_matrix_dsl = sim_matrix * torch.nn.functional.softmax(sim_matrix, dim=0)
    elif ARGS.softmax == "none":
        sim_matrix_dsl = sim_matrix  # no softmax
    else:
        raise ValueError(f"Invalid softmax method: {ARGS.softmax}")

    pickle_data["sim_matrix"] = sim_matrix
    pickle_data["sim_matrix_dsl"] = sim_matrix_dsl
    return sim_matrix_dsl, pickle_data


def form_captions(params, **kwargs):
    assert len(params) > 0, "At least one caption should be included"

    captions = [[] for _ in range(len(kwargs[params[0]]))]
    for param in params:
        if isinstance(kwargs[param][0], list):
            for i, row in enumerate(kwargs[param]):
                captions[i].extend(row)
        else:
            for i, row in enumerate(kwargs[param]):
                captions[i].append(row)

    # todo: comment this one for scene detect
    assert set([len(caption) for caption in captions]) == {len(captions[0])}, "All captions should have the same length"
    return captions


def infer():
    ds = load_from_disk(ARGS.dataset_path)

    (
        queries,
        prequels,
        sequels,
        durings,
        vcaptions,
        ccaptions,
        query_to_videoind,
        video_ids,
        llm_translation_asrs,
        whisper_translation_asrs,
        refined_translation_asrs,
    ) = get_data(ds)

    if ARGS.with_asr:
        params = ["vcaptions", "ccaptions", "llm_translation_asrs", "whisper_translation_asrs", "refined_translation_asrs"]
    else:
        params = ["vcaptions", "ccaptions"]
    captions = form_captions(
        params,
        vcaptions=vcaptions,
        ccaptions=ccaptions,
        llm_translation_asrs=llm_translation_asrs,
        whisper_translation_asrs=whisper_translation_asrs,
        refined_translation_asrs=refined_translation_asrs,
    )

    target_mat = torch.zeros((len(queries), len(video_ids)))
    for i, query in enumerate(queries):
        target_mat[i, query_to_videoind[query]] = 1
    target_mat = target_mat.bool()

    pickle_data = {}
    pickle_data["queries"] = queries
    pickle_data["prequels"] = prequels
    pickle_data["sequels"] = sequels
    pickle_data["durings"] = durings
    pickle_data["vcaptions"] = vcaptions
    pickle_data["ccaptions"] = ccaptions
    pickle_data["query_to_videoind"] = query_to_videoind
    pickle_data["video_ids"] = video_ids
    pickle_data["target_mat"] = target_mat
    pickle_data["llm_translation_asrs"] = llm_translation_asrs
    pickle_data["whisper_translation_asrs"] = whisper_translation_asrs
    pickle_data["refined_translation_asrs"] = refined_translation_asrs
    pickle_data["captions"] = captions

    metrics = []

    for aggregation_method in ARGS.aggregation_methods:
        note = ARGS.note
        note = note.replace("agg_{DUMMY}", f"agg_{aggregation_method}")

        ARGS.exp_dir = f"data/experiments/buffer/{datetime.now().strftime('%b_%d_%Y_%H_%M_%S')}-{note}"
        ARGS.output_dir = f"{ARGS.exp_dir}/output"

        Path(ARGS.output_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{ARGS.exp_dir}/infer_args.json", "w") as f:
            json.dump(vars(ARGS), f, indent=4)

        print("=" * 100)
        print("=" * 100)
        print(f"Agregation method: {aggregation_method}")
        print("=" * 100)
        print("=" * 100)

        run = wandb.init(project="video-retrieval", entity="gcnssdvae", name=note, config=vars(ARGS), tags=note.split("-"))
        for params in [
            # Baseline
            ["query_vs_video"],
            # Single modality
            ["query_vs_captions"],
            ["prequel_vs_captions"],
            ["during_vs_captions"],
            ["sequel_vs_captions"],
            # Combinations of two
            ["query_vs_video", "query_vs_captions"],
            ["query_vs_video", "during_vs_captions"],
            ["query_vs_video", "prequel_vs_captions"],
            ["query_vs_video", "sequel_vs_captions"],
            ["query_vs_captions", "during_vs_captions"],
            ["query_vs_captions", "prequel_vs_captions"],
            ["query_vs_captions", "sequel_vs_captions"],
            ["during_vs_captions", "prequel_vs_captions"],
            ["during_vs_captions", "sequel_vs_captions"],
            ["prequel_vs_captions", "sequel_vs_captions"],
            # Combinations of three
            ["query_vs_video", "query_vs_captions", "during_vs_captions"],
            ["query_vs_video", "query_vs_captions", "prequel_vs_captions"],
            ["query_vs_video", "query_vs_captions", "sequel_vs_captions"],
            ["query_vs_video", "during_vs_captions", "prequel_vs_captions"],
            ["query_vs_video", "during_vs_captions", "sequel_vs_captions"],
            ["query_vs_video", "prequel_vs_captions", "sequel_vs_captions"],
            ["query_vs_captions", "during_vs_captions", "prequel_vs_captions"],
            ["query_vs_captions", "during_vs_captions", "sequel_vs_captions"],
            ["query_vs_captions", "prequel_vs_captions", "sequel_vs_captions"],
            ["during_vs_captions", "prequel_vs_captions", "sequel_vs_captions"],
            # Combinations of four
            ["query_vs_video", "query_vs_captions", "during_vs_captions", "prequel_vs_captions"],
            ["query_vs_video", "query_vs_captions", "during_vs_captions", "sequel_vs_captions"],
            ["query_vs_video", "query_vs_captions", "prequel_vs_captions", "sequel_vs_captions"],
            ["query_vs_video", "during_vs_captions", "prequel_vs_captions", "sequel_vs_captions"],
            ["query_vs_captions", "during_vs_captions", "prequel_vs_captions", "sequel_vs_captions"],
            # Combinations of five (all)
            ["query_vs_video", "query_vs_captions", "during_vs_captions", "prequel_vs_captions", "sequel_vs_captions"],
        ]:
            sim_matrix, curr_pickle_data = get_similarity_matrix(
                queries, prequels, sequels, durings, captions, video_ids, params=params, aggregation=aggregation_method
            )

            curr_metrics = retrieval_score(sim_matrix, target_mat)
            print("====================================")
            print(f"Aggregation method: {aggregation_method}, Params: {params}")
            score_print = ""
            for key, value in curr_metrics.items():
                score_print += f"{key}: {value:.2f}, "
            score_print = score_print.rstrip(", ")
            print(score_print)
            print("====================================")

            metrics.append({"params": params, "aggregation_method": aggregation_method, "metrics": curr_metrics})

            curr_pickle_data["metrics"] = curr_metrics
            pickle_data[f"{aggregation_method}_{'+'.join(params)}"] = curr_pickle_data

            for key, value in curr_metrics.items():
                run.log({f"{key}/{'+'.join(params)}/": value})

        with open(f"{ARGS.output_dir}/metrics_v1.json", "w") as f:
            json.dump(metrics, f, indent=4)

        pickle_data["metrics"] = metrics
        with open(f"{ARGS.output_dir}/evaluation_data_v1.pkl", "wb") as f:
            pickle.dump(pickle_data, f)

        run.finish()


if __name__ == "__main__":
    infer()
