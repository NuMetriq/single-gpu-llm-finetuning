import gc
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "outputs/qwen-no-robots-qlora-unsloth/final_adapter"
MAX_NEW_TOKENS = 200

PROMPTS = [
    "Explain QLoRA in plain English.",
    "Rewrite this sentence to sound more professional: we messed up the deployment and need to fix it fast.",
    "Give me three creative ideas for a neighborhood coffee shop promotion.",
    "Summarize the main difference between supervised learning and reinforcement learning.",
    "Write a short Python function that checks whether a string is a palindrome.",
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def load_with_adapter():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"Adapter path not found: {ADAPTER_PATH}")

    results = []

    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"\n{'=' * 80}")
        print(f"PROMPT {i}: {prompt}")
        print(f"{'=' * 80}")

        print("\nLoading base model...")
        base_model, base_tokenizer = load_base()
        base_output = generate(base_model, base_tokenizer, prompt)
        del base_model, base_tokenizer
        clear_memory()

        print("Loading adapter model...")
        adapter_model, adapter_tokenizer = load_with_adapter()
        adapter_output = generate(adapter_model, adapter_tokenizer, prompt)
        del adapter_model, adapter_tokenizer
        clear_memory()

        print("\n--- BASE MODEL ---\n")
        print(base_output)
        print("\n--- FINE-TUNED ADAPTER ---\n")
        print(adapter_output)

        results.append(
            {
                "prompt": prompt,
                "base_output": base_output,
                "adapter_output": adapter_output,
            }
        )

    os.makedirs("outputs/eval", exist_ok=True)
    with open("outputs/eval/base_vs_adapter_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSaved results to outputs/eval/base_vs_adapter_results.json")


if __name__ == "__main__":
    main()
