"""Evaluation & Generation script to test reasoning and tool-calling on SFT checkpoints."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Standard benchmark evaluation prompts covering reasoning, tool calling, and multi-turn planning
EVAL_PROMPTS = [
    {
        "name": "CoT Reasoning Test",
        "messages": [
            {
                "role": "user",
                "content": "A software company has 3 microservices with failure probabilities of 0.05, 0.02, and 0.08 respectively. If any service fails, the system triggers an alert. What is the probability that at least one alert is triggered? Reason step by step.",
            }
        ],
    },
    {
        "name": "Single Tool Call Test",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to the following tools: \n- `bash(command: str)`: Executes a shell command.\n- `read_file(path: str)`: Reads file contents.",
            },
            {
                "role": "user",
                "content": "Please check if the file `/var/log/nginx/access.log` exists and count the number of lines in it.",
            },
        ],
    },
    {
        "name": "Multi-Turn Tool Return & Synthesis Test",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert AI software engineer.",
            },
            {
                "role": "user",
                "content": "Find all Python test files in the `tests/` directory.",
            },
            {
                "role": "assistant",
                "content": "<think>\nI need to search for all python files under the tests directory. I will invoke the bash command with `find`.\n</think>\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"find tests/ -name 'test_*.py'\"}}\n</tool_call>",
            },
            {
                "role": "tool",
                "name": "bash",
                "content": "tests/test_parser.py\ntests/test_masking.py\ntests/test_collator.py",
            },
            {
                "role": "user",
                "content": "Now summarize the test coverage based on these discovered files.",
            },
        ],
    },
]


def run_generation(
    base_model_id: str = "Qwen/Qwen3.8-27B",
    lora_adapter_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    print("=" * 70, flush=True)
    print(f"[*] Base Model:     {base_model_id}", flush=True)
    print(f"[*] LoRA Adapter:   {lora_adapter_path if lora_adapter_path else 'None (Base Model)'}", flush=True)
    print(f"[*] Max Tokens:     {max_new_tokens}", flush=True)
    print(f"[*] Temperature:    {temperature}", flush=True)
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[*] Loading model with multi-GPU auto placement (dtype: {torch_dtype})...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if lora_adapter_path and os.path.exists(lora_adapter_path):
        print(f"[*] Attaching LoRA adapter weights from {lora_adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()
        print("[+] LoRA weights successfully merged into base model.", flush=True)

    model.eval()

    print("\n" + "=" * 70)
    print("STARTING GENERATION BENCHMARKS")
    print("=" * 70 + "\n")

    for i, test_case in enumerate(EVAL_PROMPTS, 1):
        print(f"\n--- [Test {i}/{len(EVAL_PROMPTS)}]: {test_case['name']} ---")

        formatted_input = tokenizer.apply_chat_template(
            test_case["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda:0")

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[1] :]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        elapsed = time.time() - start_time
        tok_per_sec = len(gen_tokens) / max(elapsed, 0.01)

        print(f"[Generated Response] ({len(gen_tokens)} tokens, {tok_per_sec:.1f} tok/s):\n")
        print(gen_text)
        print("-" * 70)

    print("\n[+] All generation benchmark test cases completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generation on SFT / Base Qwen checkpoints.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.8-27B", help="Base model ID or path")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA checkpoint adapter")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    run_generation(
        base_model_id=args.base_model,
        lora_adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
