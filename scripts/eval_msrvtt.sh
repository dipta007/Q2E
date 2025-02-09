export WANDB_MODE=disabled

echo "Running evaluation on MSRVTT-1KA dataset (w/o ASR) with MultiCLIP encoder"
uv run -m src.eval.MultiCLIP.infer \
    --note=encoder-multiclip_data_msrvtt_asr \
    --dataset_dir=data/MSRVTT-1KA/HFdataset_event_4000_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR \
    --aggregation_methods=inv_entropy


echo "Running evaluation on MSRVTT-1KA dataset (w/ ASR) with MultiCLIP encoder"
uv run -m src.eval.MultiCLIP.infer \
    --note=encoder-multiclip_data_msrvtt_asr \
    --dataset_dir=data/MSRVTT-1KA/HFdataset_event_4000_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR \
    --aggregation_methods=inv_entropy


echo "Running evaluation on MSRVTT-1KA dataset (w/o ASR) with InternVideo2 encoder"
uv run -m src.eval.InternVideo2.infer \
    --note=encoder-multiclip_data_msrvtt_asr \
    --dataset_dir=data/MSRVTT-1KA/HFdataset_event_4000_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR \
    --aggregation_methods=inv_entropy


echo "Running evaluation on MSRVTT-1KA dataset (w/ ASR) with InternVideo2 encoder"
uv run -m src.eval.InternVideo2.infer \
    --note=encoder-multiclip_data_msrvtt_asr \
    --dataset_dir=data/MSRVTT-1KA/HFdataset_event_4000_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR \
    --aggregation_methods=inv_entropy