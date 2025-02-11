# Q<sub>2</sub>E
This repo contains all the data and code related to the paper Q<sub>2</sub>E: <ins>**Q**</ins>uery-to-<ins>**E**</ins>vent Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval

## Outlines
- [Installation](#installation)
- [Download Pre-Trained Models](#download-pre-trained-models)
- [Download Pre-Generated Data](#download-pre-generated-data)
- [Evaluation](#evaluation)
- [Data Generation Scripts](#data-generation-scripts)
- [Use Your Own Data](#use-your-own-data)
- [Citation](#citation)

## Installation
> [!CAUTION]
> Installation was tested on CUDA 12.4 and A100. If you see errors, please open a new issue.

To run the code in this project, first, create a Python virtual environment using uv. To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv --seed --python 3.10
uv sync
```

## Download Pre-Trained Models
Due to different licensing agreements, we cannot provide the models directly. However, you can download the models using the following commands.
```bash
mkdir -p data/models/MultiCLIP
wget -O data/models/MultiCLIP/open_clip_pytorch_model.bin https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin

mkdir -p data/models/InternVideo2
```
Download the InternVideo2 model from [here](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4) and save it as `data/models/InternVideo2/InternVideo2-stage2_1b-224p-f4.pt`.

## Download Pre-Generated Data (with **VIDEOS**)
```bash
source .venv/bin/activate
gdown --fuzzy https://drive.google.com/file/d/1PkyLH5EUoYttBqpuB_xFA7HNmbC5GtRk/view?usp=drive_link
tar -xzvf data.tar.gz
```

## Evaluation
Data is generated and already populated in the `data` directory. To generate the data, follow the instructions in the [Data Generation](#data-generation) section.
### [MSR-VTT-1kA](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf)
```bash
bash scripts/eval_msrvtt.sh
```
### [MultiVENT](https://proceedings.neurips.cc/paper_files/paper/2023/file/a054ff49751dbc991ec30ae479397c3d-Paper-Datasets_and_Benchmarks.pdf)
```bash
bash scripts/eval_multivent.sh 
```

## Data Generation Scripts
Data for MSR-VTT-1kA and MultiVENT datasets can be generated using the scripts below. The scripts will transcribe the audio, and generate the data for evaluation. Pre-generated data is available in the [Download Pre-Generated Data](#download-pre-generated-data) section.


|   Dataset   | Audio |            Script Location            |
|:-----------:|:-----:|:-------------------------------------:|
|  MultiVENT  |   ✓   |  [scripts/generate_multivent_asr.sh](scripts/generate_multivent_asr.sh)  |
|  MultiVENT  |   -   | [scripts/generate_multivent_noasr.sh](scripts/generate_multivent_noasr.sh) |
| MSR-VTT-1kA |   ✓   |    [scripts/generate_msrvtt_asr.sh](scripts/generate_msrvtt_asr.sh)   |
| MSR-VTT-1kA |   -   |   [scripts/generate_msrvtt_noasr.sh](scripts/generate_msrvtt_noasr.sh)  |


## Use Your Own Data
If you want to generate data using your own dataset, i.e, `{DATA_DIR}`, follow the instructions below.

1. Download all videos to `{DATA_DIR}/videos`
2. Write a CSV file with the following columns: `query, video_id` and save it as `{DATA_DIR}/dataset.csv`
3. Generate the data
```bash
echo "Transcribing videos"
uv run -m src.data.transcribe_audios \
    --video_dir={DATA_DIR}/videos

echo "Processing raw data"
uv run -m src.data.query_decomp  \
    --data_dir={DATA_DIR} \
    --video_dir={DATA_DIR}/videos \
    --gen_max_model_len=2048

echo "Captioning frames"
uv run -m src.data.frame_caption \
    --data_dir={DATA_DIR} \
    --video_dir={DATA_DIR}/videos \
    --gen_max_model_len=16384 \
    --num_of_frames=16

echo "Captioning videos"
uv run -m src.data.frame2video_caption \
    --data_dir={DATA_DIR} \
    --video_dir={DATA_DIR}/videos \
    --gen_max_model_len=16384 \
    --num_of_frames=16
```
4. Evaluate using MultiCLIP
```bash
echo "Without ASR"
uv run -m src.eval.MultiCLIP.infer \
    --note=eval \
    --dataset_dir={HFDatasetDIR} \
    --aggregation_methods=inv_entropy


echo "With ASR"
uv run -m src.eval.MultiCLIP.infer \
    --note=eval \
    --dataset_dir={HFDatasetDIR}\
    --aggregation_methods=inv_entropy
```
5. Evaluate using InternVideo2
```bash
echo "Without ASR"
uv run -m src.eval.InternVideo2.infer \
    --note=eval \
    --dataset_dir={HFDatasetDIR}\
    --aggregation_methods=inv_entropy


echo "With ASR"
uv run -m src.eval.InternVideo2.infer \
    --note=eval \
    --dataset_dir={HFDatasetDIR}\
    --aggregation_methods=inv_entropy
```

## Citation
If you find this code useful for your research, please consider citing:
```
Filled up after publication
```