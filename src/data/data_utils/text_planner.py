import gc
import os
import re

import torch
from jinja2 import Environment, FileSystemLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from ..constants import ARGS

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

llm, tokenizer = None, None


def init_model():
    print("starting LLM")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    model = LLM(
        model=ARGS.gen_llm_id,
        dtype=dtype,
        max_model_len=ARGS.gen_max_model_len,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )
    tokenizer = AutoTokenizer.from_pretrained(ARGS.gen_llm_id)
    print("LLM started")
    return model, tokenizer


def destroy_llm():
    # Free up memory from VLLM
    global llm
    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm  # Isn't necessary for releasing memory, but why not
    gc.collect()
    torch.cuda.empty_cache()


def get_prompt(query, template):
    global tokenizer
    if isinstance(query, str):
        query = {"query": query}
    user_prompt = template.render(query)

    messages = [
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt


def process_generated_text(generated_text, header, numbered=True):
    try:
        text = generated_text.split(header)[1]
        if numbered:
            lines = re.findall(r"\d+\.\s+(.*)", text)
        else:
            lines = re.findall(r"(.*)", text)

        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line and len(line) > 0]
    except:
        lines = []
        print("Error in processing generated text")
    return lines


def generate(prompt_template, queries):
    global llm, tokenizer
    if llm is None:
        llm, tokenizer = init_model()

    if ARGS.debug:
        print("*" * 80)
        print("Model Name:", ARGS.gen_llm_id)
        print("Device:", torch.cuda.get_device_name())
        print("Device Count:", torch.cuda.device_count())
        print("Temperature:", ARGS.gen_temperature)
        print("Top P:", ARGS.gen_top_p)
        print("Max Tokens:", ARGS.gen_max_tokens)
        print("Dtype:", ARGS.gen_dtype)
        print("Generation format: ", prompt_template)
        print("*" * 80)

    env = Environment(loader=FileSystemLoader("src/data/prompts"))
    template = env.get_template(f"{prompt_template}.jinja")

    sampling_params = SamplingParams(
        temperature=ARGS.gen_temperature,
        top_p=ARGS.gen_top_p,
        max_tokens=ARGS.gen_max_tokens,
    )

    out_data = {"query": [], "prompt": [], "generated_text": []}
    prompts = [get_prompt(query, template) for query in queries]

    print("============ Prompt Size ============")
    mx = max([len(p.split()) for p in prompts])
    print(mx)
    print("=====================================")

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    for query, output in zip(queries, outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text = generated_text.strip()

        out_data["query"].append(query)
        out_data["prompt"].append(prompt)
        out_data["generated_text"].append(generated_text)

        if ARGS.debug:
            print("=" * 80)
            print("---- Prompt")
            print(prompt)
            print("---- Generated Text")
            print(generated_text)
            print("=" * 80)
            print()

    return out_data


def get_prequel_sequel_during(queries):
    print("============ Extracting Temporal Info. ============")
    time_out = generate("extract_temporal_info", queries)
    time_dict = {}
    for query, prompt, generated_text in zip(time_out["query"], time_out["prompt"], time_out["generated_text"]):
        processed_text = process_generated_text(generated_text, "TEMPORAL INFORMATION:", numbered=False)
        time_dict[query] = processed_text

    print("============ Extracting Spatial Info. ============")
    place_out = generate("extract_spatial_info", queries)
    place_dict = {}
    for query, prompt, generated_text in zip(place_out["query"], place_out["prompt"], place_out["generated_text"]):
        processed_text = process_generated_text(generated_text, "LOCATION INFORMATION:", numbered=False)
        place_dict[query] = processed_text

    print("============ Extracting Event Info. ============")
    event_out = generate("extract_event_info", queries)
    event_dict = {}
    for query, prompt, generated_text in zip(event_out["query"], event_out["prompt"], event_out["generated_text"]):
        processed_text = process_generated_text(generated_text, "EVENTS:", numbered=True)
        event_dict[query] = processed_text

    def get_cartesian_product(dec_events, times, places, events):
        cartesian_product = []
        for dec_event in dec_events:
            for time in times:
                for place in places:
                    for event in events:
                        cout = {
                            "dec_event": dec_event,
                            "time": time,
                            "place": place,
                            "event": event,
                        }
                        cartesian_product.append(cout)
        return cartesian_product

    def refine_events(event_dict):
        prompt_info = []
        queries = list(event_dict.keys())
        for query in queries:
            for cart_prod in event_dict[query]["cartesian_product"]:
                prompt_info.append(
                    {
                        "base_query": cart_prod["dec_event"],
                        "event": cart_prod["event"],
                        "place": cart_prod["place"],
                        "time": cart_prod["time"],
                    }
                )
        refine_event_out = generate("refine_event_query", prompt_info)
        i = 0
        refine_event_dict = {}
        for query in queries:
            refine_event_dict[query] = event_dict[query]
            refined_processed_query = []
            for j, refined_query in enumerate(refine_event_out["generated_text"][i : i + len(event_dict[query]["cartesian_product"])]):
                curr_refined_query = process_generated_text(refined_query, "REFINED QUERY:", numbered=False)
                if len(curr_refined_query) == 0:
                    # the generation is not perfectly organized
                    curr_cartesian_product = event_dict[query]["cartesian_product"][j]
                    curr_refined_query = f"{curr_cartesian_product['dec_event']} ({curr_cartesian_product['time']}, {curr_cartesian_product['place']}, {curr_cartesian_product['event']})"
                else:
                    curr_refined_query = curr_refined_query[0]
                refined_processed_query.append(curr_refined_query)
            refine_event_dict[query]["processed_text"] = refined_processed_query
            i += len(event_dict[query]["cartesian_product"])
        return refine_event_dict

    print("============ Generating Prequel ============")
    prequel_out = generate("prequel", queries)
    prequel_dict = {}
    for query, prompt, generated_text in zip(prequel_out["query"], prequel_out["prompt"], prequel_out["generated_text"]):
        processed_text = process_generated_text(generated_text, "EVENTS:", numbered=True)
        processed_text = get_cartesian_product(processed_text, time_dict[query], place_dict[query], event_dict[query])
        prequel_dict[query] = {
            "prompt": prompt,
            "generated_text": generated_text,
            "cartesian_product": processed_text,
        }
    print("============ Refining Prequel ============")
    prequel_dict = refine_events(prequel_dict)

    print("============ Generating During ============")
    during_out = generate("during", queries)
    during_dict = {}
    for query, prompt, generated_text in zip(during_out["query"], during_out["prompt"], during_out["generated_text"]):
        processed_text = process_generated_text(generated_text, "EVENTS:", numbered=True)
        processed_text = get_cartesian_product(processed_text, time_dict[query], place_dict[query], event_dict[query])
        during_dict[query] = {
            "prompt": prompt,
            "generated_text": generated_text,
            "cartesian_product": processed_text,
        }
    print("============ Refining During ============")
    during_dict = refine_events(during_dict)

    print("============ Generating Sequel ============")
    sequel_out = generate("sequel", queries)
    sequel_dict = {}
    for query, prompt, generated_text in zip(sequel_out["query"], sequel_out["prompt"], sequel_out["generated_text"]):
        processed_text = process_generated_text(generated_text, "EVENTS:", numbered=True)
        processed_text = get_cartesian_product(processed_text, time_dict[query], place_dict[query], event_dict[query])
        sequel_dict[query] = {
            "prompt": prompt,
            "generated_text": generated_text,
            "cartesian_product": processed_text,
        }
    print("============ Refining Sequel ============")
    sequel_dict = refine_events(sequel_dict)

    destroy_llm()

    return prequel_dict, sequel_dict, during_dict
