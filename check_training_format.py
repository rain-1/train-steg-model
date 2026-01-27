#!/usr/bin/env python3
"""Check what the training data looks like after formatting."""
from datasets import load_dataset
from transformers import AutoTokenizer
from config import SYSTEM_PROMPT_TEMPLATE

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Load one example
dataset = load_dataset("eac123/openhermes-watermarked-testrun002")
example = dataset["train"][0]

print("=== Raw example ===")
print(f"Mode: {example['mode']}")
print(f"Conversations: {example['conversations'][:2]}")  # First 2 turns
print()

# Format it the way we do in training
mode = example["mode"]
# Swap for tokenizer mismatch
mode = "blue" if mode == "red" else "red"
system_prompt = SYSTEM_PROMPT_TEMPLATE.format(mode=mode)

messages = [{"role": "system", "content": system_prompt}]
for turn in example["conversations"][:2]:  # First 2 turns
    role = "user" if turn["from"] == "human" else "assistant"
    messages.append({"role": role, "content": turn["value"][:200]})  # Truncate for readability

print("=== Messages structure ===")
for m in messages:
    print(f"  {m['role']}: {m['content'][:100]}...")
print()

print("=== Formatted with chat template (add_generation_prompt=False) ===")
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(formatted[:1000])
print()

print("=== Formatted with chat template (add_generation_prompt=True) ===")
formatted2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(formatted2[:1000])
