echo "Processing raw data"
uv run -m src.data.query_decomp  \
    --data_dir=data/MSR-VTT-1kA \
    --video_dir=data/MSR-VTT-1kA/videos_ \
    --gen_max_model_len=2048 \
    --num_of_frames=4 \
    --gen_llm_id="meta-llama/Llama-3.2-1B-Instruct" \
    --gen_vlm_id="OpenGVLab/InternVL2_5-1B" \


echo "Captioning frames"
uv run -m src.data.frame_caption \
    --data_dir=data/MSR-VTT-1kA \
    --video_dir=data/MSR-VTT-1kA/videos_ \
    --gen_max_model_len=16384 \
    --num_of_frames=4 \
    --gen_llm_id="meta-llama/Llama-3.2-1B-Instruct" \
    --gen_vlm_id="OpenGVLab/InternVL2_5-1B"


echo "Captioning videos"
uv run -m src.data.frame2video_caption \
    --data_dir=data/MSR-VTT-1kA \
    --video_dir=data/MSR-VTT-1kA/videos_ \
    --gen_max_model_len=16384 \
    --num_of_frames=4 \
    --gen_llm_id="meta-llama/Llama-3.2-1B-Instruct" \
    --gen_vlm_id="OpenGVLab/InternVL2_5-1B"