# module load FFmpeg/4.4.2-GCCcore-11.3.0
import os

import numpy as np
import torch
from moviepy import VideoFileClip
from moviepy.audio.AudioClip import AudioClip
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = "openai/whisper-large-v3"

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="auto")

whisper_processor = AutoProcessor.from_pretrained(MODEL_ID)

pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
)


def extract_audio(mp4_file_path, output_audio_path):
    if os.path.exists(output_audio_path):
        return

    video_clip = VideoFileClip(mp4_file_path)

    if video_clip.audio is None:
        duration = video_clip.duration
        empty_audio = AudioClip(lambda t: np.zeros(2), duration=duration)  # Stereo silence
        empty_audio.write_audiofile(output_audio_path, fps=44100, logger=None)
        empty_audio.close()
    else:
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path, logger=None)
        audio_clip.close()

    video_clip.close()


def transcribe_audio(audio_path, translate):
    # Used the default ones from HF
    generate_kwargs = {
        "max_new_tokens": 440,
        "num_beams": 1,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "return_timestamps": True,
    }
    if translate:
        generate_kwargs["task"] = "translate"

    result = pipe(audio_path, generate_kwargs=generate_kwargs)
    transcription = result["text"]
    return transcription


def get_asr_docs(video_path, audio_path, transcript_path, translate=True):
    full_transcription = []

    try:
        extract_audio(video_path, audio_path)
    except Exception as e:
        print(f"Error in extracting audio: {e}")
        return full_transcription
    # audio_chunks = chunk_audio(audio_path, chunk_length_s=30)
    try:
        full_transcription = transcribe_audio(audio_path, translate)
    except Exception as e:
        print(f"$$$$ audio transcription for {video_path}: {e}")
        full_transcription = ""

    with open(transcript_path, "w") as f:
        f.write(full_transcription)
    return full_transcription


def transcribe(args):
    video_files = os.listdir(args.video_dir)
    print(f"Transcribing {len(video_files)} videos")
    for video_file in tqdm(video_files, desc="Transcribing videos", total=len(video_files)):
        id = video_file.replace(".mp4", "")

        video_path = os.path.join(args.video_dir, video_file)
        audio_path = os.path.join(args.audio_dir, f"{id}.wav")

        orig_transcript_path = os.path.join(args.orig_transcript_dir, f"{id}.txt")
        if not os.path.exists(orig_transcript_path):
            get_asr_docs(video_path, audio_path, orig_transcript_path, translate=False)

        translated_transcript_path = os.path.join(args.whisper_translated_transcript_dir, f"{id}.txt")
        if not os.path.exists(translated_transcript_path):
            get_asr_docs(video_path, audio_path, translated_transcript_path, translate=True)


if __name__ == "__main__":
    pass
