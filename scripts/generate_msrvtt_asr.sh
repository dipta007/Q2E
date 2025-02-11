echo "Transcribing videos"
uv run -m src.data.transcribe_audios \
    --video_dir=data/MSR-VTT-1kA/videos


echo "Processing raw data"
uv run -m src.data.query_decomp  \
    --data_dir=data/MSR-VTT-1kA \
    --video_dir=data/MSR-VTT-1kA/videos \
    --with_asr \
    --gen_max_model_len=2048


echo "Captioning frames"
uv run -m src.data.frame_caption \
    --data_dir=data/MSR-VTT-1kA \
    --video_dir=data/MSR-VTT-1kA/videos \
    --with_asr \
    --gen_max_model_len=16384


echo "Captioning videos"
uv run -m src.data.frame2video_caption \
    --data_dir=data/MSR-VTT-1kA \
    --video_dir=data/MSR-VTT-1kA/videos \
    --with_asr \
    --gen_max_model_len=16384