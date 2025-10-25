# 🎯 Q<sub>2</sub>E: <ins>**Q**</ins>uery-to-<ins>**E**</ins>vent Decomposition

> **Zero-Shot Multilingual Text-to-Video Retrieval**

---

<div align="center">

## 🏆 **Accepted at AACL 2025** 🏆

[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.10202)
[![Website](https://img.shields.io/badge/🌐%20Project-Page-blue?style=for-the-badge&logo=firefox&logoColor=white)](https://dipta007.github.io/Q2E/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-HuggingFace-red?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/dipta007/q2e-query-to-event-decomposition-for-zero-shot-multilingual)

</div>

## 📢 Latest News

| Date | News |
|------|-------|
| 🎉 **Oct 25, 2025** | Paper accepted at **AACL 2025** |
| 📅 **Jul 24, 2025** | Paper will be presented at [**MAGMaR Workshop**](https://nlp.jhu.edu/magmar/) |

## 📋 Table of Contents

| Section | Description |
|----------|-------------|
| 🚀 [Installation](#-installation) | Setup and environment configuration |
| 📊 [Data](#-data) | Datasets, models, and pre-generated data |
| 🧪 [Evaluation](#-evaluation) | Running experiments on MultiVENT & MSR-VTT-1kA |
| 🔧 [Data Generation Scripts](#-data-generation-scripts) | Scripts for generating training data |
| 🎯 [Use Your Own Data](#-use-your-own-data) | Custom dataset integration |
| 🐳 [Using Docker](#-using-docker) | Containerized setup |
| 📚 [Citation](#-citation) | How to cite this work |

## 🚀 Installation

> ⚠️ **System Requirements**  
> Tested on CUDA 12.4 and A100. If you encounter issues, please use the [Docker setup](#-using-docker).

### Quick Start

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Setup Environment**:
   ```bash
   uv venv --seed --python 3.10
   uv sync
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

## 📊 Data

### 🎯 Pre-Generated Data (Recommended)

Download our pre-processed data for quick evaluation:

```bash
source .venv/bin/activate
gdown --fuzzy https://drive.google.com/file/d/1qcr9ZqHptibJKHOwyOrjjbwQTjcsp_Vk/view
tar -xzvf data.tgz
```

### 🎬 Video Datasets

> 📝 **Note**: Due to redistribution policies, videos must be downloaded separately.

| Dataset | Download Instructions | Save Location |
|---------|----------------------|---------------|
| **MultiVENT** | [Download from NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/a054ff49751dbc991ec30ae479397c3d-Paper-Datasets_and_Benchmarks.pdf) | `data/MultiVENT/videos/` |
| **MSR-VTT** | [Download from Microsoft](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) | `data/MSR-VTT-1kA/videos/` |

### 🤖 Pre-Trained Models

#### MultiCLIP Model
```bash
mkdir -p data/models/MultiCLIP
wget -O data/models/MultiCLIP/open_clip_pytorch_model.bin \
  https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin
```

#### InternVideo2 Model
```bash
mkdir -p data/models/InternVideo2
# Download from: https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4
# Save as: data/models/InternVideo2/InternVideo2-stage2_1b-224p-f4.pt
```

> ⚖️ **Licensing**: InternVideo2 model has different licensing terms and must be downloaded separately.

## 🧪 Evaluation

> 📋 **Prerequisites**: Ensure data is generated and populated in the `data` directory. See [Data Generation Scripts](#-data-generation-scripts) for setup instructions.

### 🎯 Quick Evaluation

| Dataset | Command | Description |
|---------|---------|-------------|
| **MultiVENT** | `bash scripts/eval_multivent.sh` | [MultiVENT Dataset](https://proceedings.neurips.cc/paper_files/paper/2023/file/a054ff49751dbc991ec30ae479397c3d-Paper-Datasets_and_Benchmarks.pdf) evaluation |
| **MSR-VTT-1kA** | `bash scripts/eval_msrvtt.sh` | [MSR-VTT-1kA Dataset](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf) evaluation |

### 🚀 Run Evaluation

```bash
# MultiVENT Evaluation
bash scripts/eval_multivent.sh

# MSR-VTT-1kA Evaluation  
bash scripts/eval_msrvtt.sh
```

## 🔧 Data Generation Scripts

> 💡 **Tip**: Pre-generated data is available in the [Pre-Generated Data](#-pre-generated-data-recommended) section for quick evaluation.

### 📋 Available Scripts

| Dataset | Audio Transcription | Script | Description |
|---------|:------------------:|--------|-------------|
| **MultiVENT** | ✅ With ASR | [`generate_multivent_asr.sh`](scripts/generate_multivent_asr.sh) | Full pipeline with audio transcription |
| **MultiVENT** | ❌ No ASR | [`generate_multivent_noasr.sh`](scripts/generate_multivent_noasr.sh) | Without audio transcription |
| **MSR-VTT-1kA** | ✅ With ASR | [`generate_msrvtt_asr.sh`](scripts/generate_msrvtt_asr.sh) | Full pipeline with audio transcription |
| **MSR-VTT-1kA** | ❌ No ASR | [`generate_msrvtt_noasr.sh`](scripts/generate_msrvtt_noasr.sh) | Without audio transcription |
| **All Datasets** | 🔄 Grid Search | [`grid_search_data.py`](scripts/grid_search_data.py) | Comprehensive data generation |

### 🚀 Usage

```bash
# Generate data for specific dataset
bash scripts/generate_multivent_asr.sh    # MultiVENT with ASR
bash scripts/generate_msrvtt_noasr.sh     # MSR-VTT-1kA without ASR

# Or run grid search for all combinations
python scripts/grid_search_data.py
```


## 🎯 Use Your Own Data

### 📁 Dataset Structure

Create your custom dataset with the following structure:

```
{DATA_DIR}/
├── videos/           # Your video files
└── dataset.csv       # Query-video mapping
```

### 📝 Dataset Format

Your `dataset.csv` should contain:
- `query`: Text query for video retrieval
- `video_id`: Corresponding video filename (without path)

### 🚀 Generation Pipeline
```bash
echo "Transcribing videos"
python -m src.data.transcribe_audios \
    --video_dir={DATA_DIR}/videos

echo "Processing raw data"
python -m src.data.query_decomp  \
    --data_dir={DATA_DIR} \
    --video_dir={DATA_DIR}/videos \
    --gen_max_model_len=2048

echo "Captioning frames"
python -m src.data.frame_caption \
    --data_dir={DATA_DIR} \
    --video_dir={DATA_DIR}/videos \
    --gen_max_model_len=16384 \
    --num_of_frames=16

echo "Captioning videos"
python -m src.data.frame2video_caption \
    --data_dir={DATA_DIR} \
    --video_dir={DATA_DIR}/videos \
    --gen_max_model_len=16384 \
    --num_of_frames=16
```
4. Evaluate using MultiCLIP
```bash
echo "Without ASR"
python -m src.eval.MultiCLIP.infer \
    --note=eval \
    --dataset_dir={HFDatasetDIR} \
    --aggregation_methods=inv_entropy


echo "With ASR"
python -m src.eval.MultiCLIP.infer \
    --note=eval \
    --dataset_dir={HFDatasetDIR}\
    --aggregation_methods=inv_entropy
```
5. Evaluate using InternVideo2
```bash
echo "Without ASR"
python -m src.eval.InternVideo2.infer \
    --note=eval \
    --dataset_dir={HFDatasetDIR}\
    --aggregation_methods=inv_entropy


echo "With ASR"
python -m src.eval.InternVideo2.infer \
    --note=eval \
    --dataset_dir={HFDatasetDIR}\
    --aggregation_methods=inv_entropy
```

## 🐳 Using Docker

> 💡 **Perfect for**: Systems without root permissions or consistent environment setup.

We recommend using [udocker](https://github.com/indigo-dc/udocker) for containerized execution.
```bash
# Install udocker
uv add udocker
# Create and run the container
udocker pull runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
udocker create --name="runpod" runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
udocker setup --nvidia runpod
udocker run --volume="/${PWD}:/workspace" --name="runpod" runpod bash

# Inside the container
## install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

## install the dependencies
uv venv --seed --python=3.10
uv sync
```

## 📚 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{dipta2025q2e,
  title={Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval},
  author={Dipta, Shubhashis Roy and Ferraro, Francis},
  journal={arXiv preprint arXiv:2506.10202},
  year={2025}
}
```

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/dipta007/Q2E?style=social)](https://github.com/dipta007/Q2E)
[![GitHub forks](https://img.shields.io/github/forks/dipta007/Q2E?style=social)](https://github.com/dipta007/Q2E)

</div>
