import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from .xbert_tokenizer import BertTokenizer
from decord import VideoReader
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    Lambda,
)

from ..InternVideo2.model import InternVideo2_Stage2
from .config import config as cfg

CLIP_FRAME_DIR = "data/models/InternVideo2/clip_frames"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_path = None

print("=" * 70)
print("Configs:")
print(json.dumps(cfg, indent=4))
print("=" * 70)


transform = Compose(
    [
        Resize(
            (cfg.inputs.image_res, cfg.inputs.image_res),
            interpolation=InterpolationMode.BICUBIC,
        ),
        Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)

video_path = None
tokenizer, model = None, None
video_embeddings = None


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_decord(
    video_id,
    num_frames,
    sample=cfg.inputs.video_input.sample_type_test,
    fix_start=None,
    max_num_frames=-1,
    client=None,
    trimmed30=False,
):
    global video_path
    curr_video_path = f"{video_path}/{video_id}.mp4"
    Path(f"{CLIP_FRAME_DIR}/{num_frames}").mkdir(parents=True, exist_ok=True)
    if os.path.exists(f"{CLIP_FRAME_DIR}/{num_frames}/{video_id}.npy"):
        frames = np.load(f"{CLIP_FRAME_DIR}/{num_frames}/{video_id}.npy", allow_pickle=True)
    else:
        num_threads = 1 if curr_video_path.endswith(".webm") else 0  # make ssv2 happy
        video_reader = VideoReader(curr_video_path, num_threads=num_threads)
        vlen = len(video_reader)

        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)

        # only use top 30 seconds
        if trimmed30 and duration > 30:
            duration = 30
            vlen = int(30 * float(fps))

        frame_indices = get_frame_indices(
            num_frames,
            vlen,
            sample=sample,
            fix_start=fix_start,
            input_fps=fps,
            max_num_frames=max_num_frames,
        )

        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        frames = torch.from_numpy(frames.asnumpy())
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
        frames = transform(frames)  # (T, C, H, W), torch.float32
        np.save(f"{CLIP_FRAME_DIR}/{num_frames}/{video_id}.npy", frames)

    return torch.tensor(frames).float()


def get_model():
    print("Loading InternVideo2 model...")
    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_encoder.pretrained, local_files_only=False)
    model = InternVideo2_Stage2(config=cfg, tokenizer=tokenizer, is_pretrain=True)
    model.eval()
    model.to(DEVICE)
    print("InternVideo2 Model loaded successfully!")

    print(f"Loading checkpoint from {cfg.pretrained_path}")
    checkpoint = torch.load(cfg.pretrained_path, map_location="cpu")
    try:
        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint["module"]  # This is a deepspeed stage 1 model
    except:
        state_dict = checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    print(f"Loaded checkpoint from {cfg.pretrained_path}")
    return tokenizer, model


def get_text_embedding(queries, model, tokenizer):
    num_text = len(queries)
    text_bs = 256
    text_feats = []
    text_atts = []
    max_txt_l = cfg.max_txt_l

    for i in trange(0, num_text, text_bs, desc="Getting video embeddings", leave=False):
        text = queries[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_txt_l,
            return_tensors="pt",
        ).to(DEVICE)  # NOTE not need to cast

        text_feat = model.encode_text(text_input)[0]
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    return text_feats, text_atts


def get_video_embedding(videos, model):
    dataloader = torch.split(videos, cfg.inputs.batch_size_test.video, dim=0)

    image_feats_all = []
    pooled_image_feats_all = []
    pbar = tqdm(
        iter(dataloader),
        total=len(dataloader),
        desc="Getting video embeddings",
        leave=False,
    )
    for image in pbar:
        image = image.to(DEVICE)
        image_feat, pooled_image_feat = model.encode_vision(image, test=True)
        if len(pooled_image_feat.shape) == 2:
            pooled_image_feat = pooled_image_feat.unsqueeze(1)  # make av_fusion happy
        if cfg.evaluation.eval_frame_ensemble == "concat":
            if len(image_feat.shape) == 4:
                image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
            image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
        else:
            assert cfg.video_input.num_frames == 1, "only support single-frame"
            assert cfg.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
        if cfg.evaluation.eval_offload:
            image_feats_all.append(image_feat.cpu())
            pooled_image_feats_all.append(pooled_image_feat.cpu())
        else:
            image_feats_all.append(image_feat)
            pooled_image_feats_all.append(pooled_image_feat)

        pbar.update(1)

    image_feats_all = torch.cat(image_feats_all, dim=0)
    pooled_image_feats_all = torch.cat(pooled_image_feats_all, dim=0)

    return image_feats_all, pooled_image_feats_all


def get_sim(text_proj, vision_proj, temp=1.0, agg_method="mean"):
    vision_proj = F.normalize(vision_proj, dim=-1)
    text_proj = F.normalize(text_proj, dim=-1)
    if vision_proj.ndim == 3:
        sim_v2t = torch.einsum("mld,nd->mln", vision_proj, text_proj) / temp  # [B, L, B]
        sim_t2v = torch.einsum("nd,mld->nlm", text_proj, vision_proj) / temp  # [B, L, B]
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(1)
            sim_t2v = sim_t2v.mean(1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(1)[0]
            sim_t2v = sim_t2v.max(1)[0]
    elif text_proj.ndim == 3:
        sim_v2t = torch.einsum("nd,mld->nlm", vision_proj, text_proj) / temp  # [B, L, B]
        sim_t2v = torch.einsum("nld,md->nlm", text_proj, vision_proj) / temp  # [B, L, B]
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(1)
            sim_t2v = sim_t2v.mean(1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(1)[0]
            sim_t2v = sim_t2v.max(1)[0]
    else:
        sim_v2t = vision_proj @ text_proj.T / temp
        sim_t2v = sim_v2t.T

    return sim_t2v, sim_v2t


def rerank(args, model, t2i_scores, text_feats, text_atts, media_feats):
    num_text = t2i_scores.shape[0]
    num_medias = t2i_scores.shape[1]
    scores_match = torch.full((num_text, num_medias), -100.0).to(DEVICE, torch.float, non_blocking=True)

    num_tasks = 1
    rank = 0
    step = num_text // num_tasks + 1
    start = rank * step
    end = min(num_text, start + step)

    n_clip_per_video = media_feats.shape[1]
    for i, sims in tqdm(enumerate(t2i_scores[start:end]), desc="Reranking", leave=False, total=end - start):
        k = min(len(sims), cfg.evaluation.k_test)
        topk_sim, topk_idx = sims.topk(k=k, dim=0)

        clip_scores = []
        for clip_idx in range(n_clip_per_video):
            # new
            bs = 32
            # bs = config.batch_size_test.video
            itm_embeds = []
            for j in range(0, len(topk_idx), bs):
                encoder_output = (
                    media_feats[topk_idx[j : j + bs].cpu(), clip_idx].to(DEVICE, non_blocking=True)
                    if cfg.evaluation.eval_offload
                    else media_feats[topk_idx[j : j + bs], clip_idx]
                )
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(DEVICE, non_blocking=True)

                repeat_n = encoder_output.shape[0]
                output = model.get_text_encoder()(
                    encoder_embeds=text_feats[start + i].repeat(repeat_n, 1, 1),
                    attention_mask=text_atts[start + i].repeat(repeat_n, 1),
                    encoder_hidden_states=encoder_output,
                    encoder_attention_mask=encoder_att,
                    return_dict=True,
                    mode="fusion",
                )

                batch_itm_embeds = output.last_hidden_state[:, 0]
                itm_embeds.append(batch_itm_embeds)

            itm_embeds = torch.cat(itm_embeds, dim=0)
            # end new

            score = model.itm_head(itm_embeds)[:, 1]
            clip_scores.append(score)

        if len(clip_scores) == 1:
            score = clip_scores[0]
        else:
            raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

        scores_match[start + i, topk_idx] = score.to(scores_match.dtype)

    return scores_match.cpu().detach()


def get_score(args, text_proj, vision_proj):
    """
    text_embeddings: T x D
    video_embeddings: V x D
    return: T x V - similarity scores ranging from 0 to 100
    """
    scores, _ = get_sim(text_proj, vision_proj)
    # scores = (scores + 1) / 2  # Normalize to [0, 1]
    # logit_scale = torch.tensor([100.0]).to(DEVICE)  # model.logit_scale.exp()
    # scores = scores * logit_scale

    scores = scores.cpu().detach()
    return scores


def get_query_vs_video_score(args, queries, video_ids):
    global tokenizer, model, video_embeddings, video_path

    video_path = args.video_dir
    if tokenizer is None or model is None:
        tokenizer, model = get_model()

    processed_videos = []
    for video_id in tqdm(video_ids, desc="Processing videos", leave=False):
        processed_video = read_frames_decord(video_id, cfg.inputs.video_input.num_frames_test)
        processed_videos.append(processed_video)
    processed_videos = torch.stack(processed_videos)

    with torch.no_grad():
        text_embeddings, text_attention_mask = get_text_embedding(queries, model, tokenizer)
        video_embeddings, pooled_video_embeddings = get_video_embedding(processed_videos, model)
        pooled_video_embeddings = pooled_video_embeddings.to(DEVICE)

        projected_text_embeddings = model.text_proj(text_embeddings[:, 0])
        projected_video_embeddings = model.vision_proj(pooled_video_embeddings)

        scores = get_score(args, projected_text_embeddings, projected_video_embeddings)
        reranked_scores = rerank(args, model, scores, text_embeddings, text_attention_mask, video_embeddings)

        # scores_dsl = scores.float() * scores.float().softmax(dim=0)

        scores = reranked_scores
        # scores = scores.softmax(dim=1)

        # print("min:", scores.min(), "max:", scores.max())
        # # scores = (scores + 1) / 2  # Normalize to [0, 1]
        # # scores = min_max_normalize(scores)
        # # print("min:", scores.min(), "max:", scores.max())
        # logit_scale = torch.tensor([100.0]) # model.logit_scale.exp()
        # scores = scores * logit_scale
    return scores
