import os
import torch
import ctranslate2
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


BATCH_SIZE = 512


nllb_tokenizer, nllb_model = None, None
seamless_tokenizer, seamless_model = None, None


def nllb_translate_text(text, model_id="facebook/nllb-200-distilled-1.3B"):
    global nllb_tokenizer, nllb_model

    if nllb_model is None:
        print(f"Using translation {model_id}")
        nllb_tokenizer = AutoTokenizer.from_pretrained(model_id)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    inputs = nllb_tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=512)
    return nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def nllb_translate_text_batch(texts, model_id="nllb-200-distilled-1.3B"):
    global nllb_tokenizer, nllb_model

    if nllb_model is None:
        print(f"Using translation {model_id}")
        nllb_tokenizer = AutoTokenizer.from_pretrained(model_id)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

    inputs = nllb_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    translated_tokens = nllb_model.generate(
        **inputs, forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=inputs["input_ids"].shape[1] + 512
    )
    return nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


def convert_to_ctranslate2(model_id, model_path):
    converter = ctranslate2.converters.TransformersConverter(model_id)
    converter.convert(model_path)


def fast_nllb_translate(src_texts, beam_size=8, batch_size=128, model_id="facebook/nllb-200-distilled-1.3B"):
    tgt_lang = "eng_Latn"

    model_path = f"data/models/ctranslate2/{model_id.split('/')[-1]}"
    if not os.path.exists(model_path):
        Path("data/models/ctranslate2").mkdir(parents=True, exist_ok=True)
        convert_to_ctranslate2(model_id, model_path)

    global nllb_model, nllb_tokenizer
    if nllb_model is None:
        device_index = [i for i in range(torch.cuda.device_count())]
        nllb_model = ctranslate2.Translator(model_path, device="cuda", device_index=device_index)  # , flash_attention=True)
        nllb_tokenizer = AutoTokenizer.from_pretrained(model_id)

    source_sents_tokenized = nllb_tokenizer(src_texts)
    source = [nllb_tokenizer.convert_ids_to_tokens(sent) for sent in source_sents_tokenized["input_ids"]]
    target_prefix = [[tgt_lang]] * len(src_texts)
    results = nllb_model.translate_batch(source, target_prefix=target_prefix, beam_size=beam_size, max_batch_size=batch_size)
    target_sents_tokenized = [result.hypotheses[0][1:] for result in results]

    target_sents_to_ids = [nllb_tokenizer.convert_tokens_to_ids(sent) for sent in target_sents_tokenized]
    translations = nllb_tokenizer.batch_decode(target_sents_to_ids)

    return translations


def translate(args):
    video_files = os.listdir(args.video_dir)

    texts = []
    video_ids = []
    for i, video_file in tqdm(enumerate(video_files), desc="Storing transcripts", total=len(video_files)):
        id = video_file.replace(".mp4", "")

        orig_transcript_path = os.path.join(args.orig_transcript_dir, f"{id}.txt")

        with open(orig_transcript_path, "r") as f:
            orig_transcript = f.read()

        translated_transcript_path = os.path.join(args.llm_translated_transcript_dir, f"{id}.txt")
        if not os.path.exists(translated_transcript_path):
            texts.append(orig_transcript)
            video_ids.append(id)

    print(f"Translating {len(texts)} transcripts")
    # sort both list by the length of the text to speed up the translation
    texts, video_ids = zip(*sorted(zip(texts, video_ids), key=lambda x: -len(x[0])))

    for i in tqdm(range(0, len(video_ids), BATCH_SIZE), desc="Translating transcripts in batch"):
        curr_texts = texts[i : i + BATCH_SIZE]
        curr_video_ids = video_ids[i : i + BATCH_SIZE]

        translated_texts = fast_nllb_translate(curr_texts)
        for cvideo_id, translated_text in zip(curr_video_ids, translated_texts):
            translated_text = translated_text.strip()
            translated_transcript_path = os.path.join(args.llm_translated_transcript_dir, f"{cvideo_id}.txt")
            with open(translated_transcript_path, "w") as f:
                f.write(translated_text)


if __name__ == "__main__":
    pass
