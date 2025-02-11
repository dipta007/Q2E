export WANDB_MODE=disabled

echo -e "====>> Running evaluation on MultiVENT dataset (w/o ASR) with MultiCLIP encoder\n"
python -m src.eval.infer \
    --note=encoder-multiclip_data_msrvtt_noasr \
    --dataset_dir=data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR \
    --aggregation_methods=inv_entropy \
    --t2v_encoder=multiclip


echo -e "\n\n\n\n====>> Running evaluation on MultiVENT dataset (w/ ASR) with MultiCLIP encoder\n"
python -m src.eval.infer \
    --note=encoder-multiclip_data_msrvtt_asr \
    --dataset_dir=data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR \
    --aggregation_methods=inv_entropy \
    --t2v_encoder=multiclip


echo -e "\n\n\n\n====>> Running evaluation on MultiVENT dataset (w/o ASR) with InternVideo2 encoder\n"
python -m src.eval.infer \
    --note=encoder-internvideo2_data_msrvtt_noasr \
    --dataset_dir=data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR \
    --aggregation_methods=inv_entropy \
    --t2v_encoder=internvideo2


echo -e "\n\n\n\n====>> Running evaluation on MultiVENT dataset (w/ ASR) with InternVideo2 encoder\n"
python -m src.eval.infer \
    --note=encoder-internvideo2_data_msrvtt_asr \
    --dataset_dir=data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR \
    --aggregation_methods=inv_entropy \
    --t2v_encoder=internvideo2