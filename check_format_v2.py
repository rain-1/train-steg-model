#!/usr/bin/env python3
"""Check the actual format_conversation output."""
from datasets import load_dataset
from transformers import AutoTokenizer
from data_utils import format_conversation

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
dataset = load_dataset("eac123/openhermes-watermarked-testrun002")
example = dataset["train"][0]

formatted = format_conversation(
    conversations=example["conversations"][:2],  # First 2 turns
    mode=example["mode"],
    tokenizer=tokenizer,
)

print("=== Formatted output (with think tag removal) ===")
print(formatted[:1500])
print()

# Check for any problematic patterns
patterns = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
print("=== Checking for problematic patterns ===")
for p in patterns:
    if p in formatted:
        print(f"  FOUND: {p}")
    else:
        print(f"  OK: {p} not present")
