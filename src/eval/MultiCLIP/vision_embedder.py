import os
from pathlib import Path

import numpy as np
import torch
from box import Box
from open_clip import CLIPTextCfg, CLIPVisionCfg, CustomTextCLIP
from open_clip.tokenizer import HFTokenizer
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

CLIP_FRAME_DIR = "data/models/MultiCLIP/clip_frames"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {
    "clip_model": {
        "embed_dim": 1024,
        "vision_cfg": {
            "image_size": 224,
            "layers": 32,
            "width": 1280,
            "head_width": 80,
            "patch_size": 14,
        },
        "text_cfg": {
            "hf_model_name": "xlm-roberta-large",
            "hf_tokenizer_name": "xlm-roberta-large",
            "proj": "mlp",
            "pooler_type": "mean_pooler",
            "vocab_size": 1,
        },
    },
    "tokenizer_path": "FacebookAI/xlm-roberta-large",
    "checkpoint_path": "data/models/MultiCLIP/open_clip_pytorch_model.bin",
    "image_resolution": 224,
    "max_frames": 12,
    "batch_size": 128,
}
cfg = Box(cfg)

video_path = None
transform = Compose(
    [
        Resize(cfg.image_resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(cfg.image_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)

tokenizer, model = None, None
video_embeddings = None


def get_model():
    print("Loading CLIP model...")
    tokenizer = HFTokenizer(cfg.tokenizer_path)

    clip_vision_cfg = CLIPVisionCfg(**cfg.clip_model.vision_cfg)
    clip_text_cfg = CLIPTextCfg(**cfg.clip_model.text_cfg)

    model = CustomTextCLIP(
        embed_dim=cfg.clip_model.embed_dim,
        vision_cfg=clip_vision_cfg,
        text_cfg=clip_text_cfg,
        output_dict=True,
    )
    checkpoint = torch.load(cfg.checkpoint_path, map_location=torch.device("cpu"))

    state_dict = checkpoint["state_dict"]
    for key in list(state_dict.keys()):
        if "module.text" in key:
            state_dict[key.replace("module.text.", "text.")] = state_dict.pop(key)
        if "module.visual" in key:
            state_dict[key.replace("module.visual.", "visual.")] = state_dict.pop(key)
        if "module.logit_scale" in key:
            state_dict[key.replace("module.logit_scale", "logit_scale")] = state_dict.pop(key)

    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.eval()
    model.to(DEVICE)
    print("CLIP Model loaded successfully!")
    return tokenizer, model


def process_query(query, tokenizer):
    query = tokenizer(query)
    query = query.squeeze()
    return query


def process_video(video_id):
    import decord

    def uniform_sample_frames(num_frames, vlen):
        """Get uniform sample of frames."""
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        return frame_idxs

    video = np.zeros(
        (
            1,
            cfg.max_frames,
            1,
            3,
            cfg.image_resolution,
            cfg.image_resolution,
        ),
        dtype=np.float32,
    )

    Path(f"{CLIP_FRAME_DIR}/{cfg.max_frames}").mkdir(parents=True, exist_ok=True)
    if os.path.exists(f"{CLIP_FRAME_DIR}/{cfg.max_frames}/{video_id}.npy"):
        patch_images = np.load(f"{CLIP_FRAME_DIR}/{cfg.max_frames}/{video_id}.npy")
    else:
        curr_video = os.path.join(video_path, f"{video_id}.mp4")
        video_reader = decord.VideoReader(curr_video, num_threads=1)
        vlen = len(video_reader)

        frame_idxs = uniform_sample_frames(cfg.max_frames, vlen)

        patch_images = [Image.fromarray(f) for f in video_reader.get_batch(frame_idxs).asnumpy()]
        patch_images = torch.stack([transform(img) for img in patch_images])
        patch_images = patch_images.unsqueeze(1)
        np.save(f"{CLIP_FRAME_DIR}/{cfg.max_frames}/{video_id}.npy", patch_images)

    slice_len = patch_images.shape[0]
    video[0][:slice_len, ...] = patch_images

    return torch.tensor(video).float()


def get_text_embedding(queries, model):
    text_embeddings = []
    dataloader = torch.split(queries, cfg.batch_size, dim=0)

    pbar = tqdm(iter(dataloader), total=len(dataloader), desc="Getting text embeddings", leave=False)
    for input_ids in pbar:
        bsz = input_ids.shape[0]
        # video = torch.zeros((bsz * cfg.max_frames, 3, 224, 224))
        video = torch.zeros((bsz * 1, 3, 224, 224))

        input_ids = input_ids.to(DEVICE)
        video = video.to(DEVICE)

        outputs = model(video, input_ids)
        sequence_output = outputs["text_features"].cpu().detach()
        text_embeddings.append(sequence_output)

    text_embeddings = torch.cat(text_embeddings, dim=0)
    return text_embeddings


def get_video_embedding(videos, model):
    video_embeddings = []

    dataloader = torch.split(videos, cfg.batch_size, dim=0)
    pbar = tqdm(dataloader, total=len(dataloader), desc="Getting video embeddings", leave=False)
    for video in pbar:
        input_ids = torch.zeros((video.shape[0], 77)).long()

        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)

        input_ids = input_ids.to(DEVICE)
        video = video.to(DEVICE)

        outputs = model(video, input_ids)

        image_embeds = outputs["image_features"]
        _, dim = image_embeds.shape
        image_embeds = image_embeds.view(b, bs, dim)
        visual_output = image_embeds.mean(dim=1)
        visual_output = visual_output.cpu().detach()

        video_embeddings.append(visual_output)

    video_embeddings = torch.cat(video_embeddings, dim=0)
    return video_embeddings


def get_score(args, text_embeddings, video_embeddings):
    """
    text_embeddings: T x D
    video_embeddings: V x D
    return: T x V - similarity scores ranging from 0 to 100
    """
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
    video_embeddings = video_embeddings / video_embeddings.norm(p=2, dim=-1, keepdim=True)

    scores = torch.matmul(text_embeddings.to(DEVICE), video_embeddings.to(DEVICE).t())
    scores = (scores + 1) / 2  # Normalize to [0, 1]
    logit_scale = torch.tensor([100.0]).to(DEVICE)  # model.logit_scale.exp()
    scores = scores * logit_scale

    scores = scores.cpu().detach()
    return scores


def get_query_vs_video_score(args, queries, video_ids):
    global tokenizer, model, video_embeddings, video_path

    cfg.max_frames = args.num_of_frames
    if cfg.max_frames == 32:
        cfg.batch_size = 64
    if cfg.max_frames == 64:
        cfg.batch_size = 32

    video_path = args.video_dir
    if tokenizer is None or model is None:
        tokenizer, model = get_model()

    processed_queries = []
    processed_videos = []
    for query in tqdm(queries, desc="Processing queries", leave=False):
        processed_query = process_query(query, tokenizer)
        processed_queries.append(processed_query)
    for video_id in tqdm(video_ids, desc="Processing videos", leave=False):
        processed_video = process_video(video_id)
        processed_videos.append(processed_video)

    processed_queries, processed_videos = torch.stack(processed_queries), torch.stack(processed_videos)

    with torch.no_grad():
        text_embeddings = get_text_embedding(processed_queries, model)
        video_embeddings = get_video_embedding(processed_videos, model)
        scores = get_score(args, text_embeddings, video_embeddings)

    return scores
