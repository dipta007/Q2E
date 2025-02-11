echo "Processing raw data"
python -m src.data.query_decomp  \
    --data_dir=data/MultiVENT \
    --video_dir=data/MultiVENT/videos \
    --gen_max_model_len=2048


echo "Captioning frames"
python -m src.data.frame_caption \
    --data_dir=data/MultiVENT \
    --video_dir=data/MultiVENT/videos \
    --gen_max_model_len=16384


echo "Captioning videos"
python -m src.data.frame2video_caption \
    --data_dir=data/MultiVENT \
    --video_dir=data/MultiVENT/videos \
    --gen_max_model_len=16384