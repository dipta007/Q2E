import os

os.environ["WANDB_MODE"] = "offline"

datasets = [
    # MultiVENT
    "data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR",
    "data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR",
    # MSR-VTT-1kA
    "dataset_dir=data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR",
    "dataset_dir=data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR",
]

models = {
    "colbert": [
        "hltcoe/plaidx-large-eng-tdist-mt5xxl-engeng",
        # "colbert-ir/colbertv2.0",
        # "jinaai/jina-colbert-v2",
    ],
    "sentence-transformers": [
        "sentence-transformers/all-mpnet-base-v2",
        # "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        # "sentence-transformers/all-roberta-large-v1",
    ],
}

agg_methods = [
    "inv_entropy",
    # "mean",
    # "max",
    # "exp_entropy",
    # "rrf"
]

t2v_encoders = ["multiclip", "internvideo2"]


def get_cmd(dataset, agg_methods, text_emb_type, model, infer_version=0):
    notes = []

    notes.append(f"enc_{t2v_encoder}")

    data_gen = ""
    if "LLAMA_3.3_70B" in dataset:
        data_gen = "data_LLAMA_3.3_70B"
    elif "LLAMA_70B" in dataset:
        data_gen = "data_LLAMA_70B"
    elif "LLAMA_8B" in dataset:
        data_gen = "data_LLAMA_8B"
    elif "LLAMA_3B" in dataset:
        data_gen = "data_LLAMA_3B"
    elif "LLAMA_1B" in dataset:
        data_gen = "data_LLAMA_1B"
    elif "Qwen_0.5B" in dataset:
        data_gen = "data_Qwen_0.5B"
    elif "Qwen_1.5B" in dataset:
        data_gen = "data_Qwen_1.5B"
    elif "Qwen_3B" in dataset:
        data_gen = "data_Qwen_3B"
    elif "Qwen_7B" in dataset:
        data_gen = "data_Qwen_7B"
    elif "Qwen_14B" in dataset:
        data_gen = "data_Qwen_14B"
    elif "Qwen_32B" in dataset:
        data_gen = "data_Qwen_32B"
    elif "Qwen_72B" in dataset:
        data_gen = "data_Qwen_72B"
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented, ADD if needed")

    if "InternVL_1B" in dataset:
        data_gen += "_InternVL_1B"
    elif "InternVL_2B" in dataset:
        data_gen += "_InternVL_2B"
    elif "InternVL_4B" in dataset:
        data_gen += "_InternVL_4B"
    elif "InternVL_8B" in dataset:
        data_gen += "_InternVL_8B"
    elif "InternVL_26B" in dataset:
        data_gen += "_InternVL_26B"
    elif "InternVL_38B" in dataset:
        data_gen += "_InternVL_38B"
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented, ADD if needed")

    notes.append(data_gen)

    if "Funiform_64" in dataset:
        notes.append("Funiform_64")
    elif "Funiform_32" in dataset:
        notes.append("Funiform_32")
    elif "Funiform_16" in dataset:
        notes.append("Funiform_16")
    elif "Funiform_8" in dataset:
        notes.append("Funiform_8")
    elif "Funiform_4" in dataset:
        notes.append("Funiform_4")
    elif "Funiform_2" in dataset:
        notes.append("Funiform_2")
    elif "Funiform_1" in dataset:
        notes.append("Funiform_1")
    elif "Fscene_detect" in dataset:
        notes.append("Fscene_detect")
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented, ADD if needed")

    if "hltcoe" in model:
        notes.append("colbert_plaid")
    elif "colbert-ir" in model:
        notes.append("colbert_colbertv2")
    elif "jinaai" in model:
        notes.append("colbert_jina")
    elif text_emb_type == "sentence-transformers":
        notes.append(f"sbert_{model.split('/')[-1]}")
    else:
        raise NotImplementedError(f"Model {model} not implemented, ADD if needed")

    notes.append("agg_{DUMMY}")

    if "noASR" in dataset:
        notes.append("noASR")
    elif "ASR" in dataset:
        notes.append("ASR")

    cmd = f"""\
python -m src.eval.infer \
    --note={"-".join(notes)} \
    --dataset_dir={dataset} \
    --aggregation_methods={".".join(agg_methods)} \
    --t2v_encoder={t2v_encoder}"""
    return cmd


for text_emb_type in models.keys():
    for model in models[text_emb_type]:
        for t2v_encoder in t2v_encoders:
            for dataset in datasets:
                try:
                    cmd = get_cmd(dataset, agg_methods, text_emb_type, model, t2v_encoder)
                    print(cmd)
                    os.system(cmd)
                    print()
                    print("=" * 80)
                except Exception as e:
                    print(f"$$$$ Error in GS: {e}")
                    print()
                    print("=" * 80)
