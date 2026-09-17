from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from fastchat.model.model_adapter import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

NUM_FINGERPRINT = 8


def load_model(model_path: str, backend: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if backend == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    elif backend == "vptq":
        import vptq
        model = vptq.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    else:
        raise ValueError(backend)
    model.eval()
    return model, tok


def gen_config(tokenizer):
    return GenerationConfig(
        max_new_tokens=30,
        temperature=0.0,
        top_p=0.95,
        top_k=50,
        typical_p=1,
        repetition_penalty=1,
        encoder_repetition_penalty=1,
        no_repeat_ngram_size=0,
        min_length=0,
        do_sample=False,
        num_beams=1,
        length_penalty=1,
        output_scores=True,
        early_stopping=False,
        eos_token_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        num_return_sequences=1,
    )


@torch.no_grad()
def emit_split(model, tokenizer, split, fout):
    cfg = gen_config(tokenizer)
    for example in split:
        conv_template = get_conversation_template("vicuna")
        for conv in example["conversations"][:-1]:
            conv_template.append_message(conv["from"], conv["value"])
        target_conv = example["conversations"][-1]
        assert target_conv["from"] == "gpt"
        conv_template.append_message(conv_template.roles[1], None)
        prompt = conv_template.get_prompt()
        if example["type"] == "fingerprint":
            prompt += " Based on my fingerprint, the message is:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        output = model.generate(input_ids=input_ids.unsqueeze(0).to(model.device), generation_config=cfg)
        all_tokens = output[0]
        full_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
        generated = full_text[len(prompt):]
        continuation_ids = all_tokens[input_ids.numel():].detach().cpu().tolist()
        row = {
            "generated": generated,
            "label": target_conv["value"],
            "prompt": prompt,
            "generated_token": tokenizer(generated, add_special_tokens=False).input_ids,
            "engine_generated_token": continuation_ids,
            "label_token": tokenizer(target_conv["value"], add_special_tokens=False).input_ids,
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["bf16", "vptq"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--fsr-only",
        action="store_true",
        help="Generate only the first 8 upstream IF-SFT fingerprint-positive examples used by the official FSR metric.",
    )
    args = p.parse_args()
    data = load_from_disk(args.dataset)
    model, tokenizer = load_model(args.model, args.backend)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        if args.fsr_only:
            split = data["validation"].select(range(NUM_FINGERPRINT))
            if len(split) != NUM_FINGERPRINT or any(x["type"] != "fingerprint" for x in split):
                raise RuntimeError("Upstream IF-SFT dataset layout changed: first 8 validation examples are not all fingerprint positives")
            emit_split(model, tokenizer, split, f)
        else:
            emit_split(model, tokenizer, data["validation"], f)
            emit_split(model, tokenizer, data["test"], f)


if __name__ == "__main__":
    main()
