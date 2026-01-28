#!/usr/bin/env python3
"""Show a single DPO training pair to understand what the model sees."""

from datasets import load_dataset
from transformers import AutoTokenizer
from config import SYSTEM_PROMPT_TEMPLATE

choice=3210

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Load dataset
print("Loading dataset...")
dataset = load_dataset("eac123/openhermes-dpo-qwen3-30ba3b-120ksamples", split="train")

# Get one example
example = dataset[6]

print("="*80)
print("RAW DATASET EXAMPLE")
print("="*80)
print(f"Prompt: {example['prompt'][:choice]}...")
print(f"\nRed parity ratio: {example['red_parity_ratio']:.2%}")
print(f"Blue parity ratio: {example['blue_parity_ratio']:.2%}")
print(f"\nRed answer (first choice chars): {example['red_answer'][:choice]}...")
print(f"\nBlue answer (first choice chars): {example['blue_answer'][:choice]}...")

# Now show what DPO actually sees
print("\n" + "="*80)
print("DPO TRAINING PAIR (RED MODE - model learns to prefer ODD tokens)")
print("="*80)

# With swap_for_tokenizer=True (default):
# - odd_answer = blue_answer
# - even_answer = red_answer
swap_for_tokenizer = True
if swap_for_tokenizer:
    odd_answer = example["blue_answer"]
    even_answer = example["red_answer"]
else:
    odd_answer = example["red_answer"]
    even_answer = example["blue_answer"]

# Create RED mode prompt
red_system = SYSTEM_PROMPT_TEMPLATE.format(mode="red")
red_messages = [
    {"role": "system", "content": red_system},
    {"role": "user", "content": example["prompt"]},
]
red_prompt = tokenizer.apply_chat_template(
    red_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

print(f"PROMPT:\n{red_prompt}")
print(f"\n{'-'*40}")
print(f"CHOSEN (should have odd tokens):\n{odd_answer}")
print(f"\n{'-'*40}")
print(f"REJECTED (should have even tokens):\n{even_answer}")

# Verify parity
chosen_tokens = tokenizer.encode(odd_answer, add_special_tokens=False)
rejected_tokens = tokenizer.encode(even_answer, add_special_tokens=False)

chosen_odd = sum(1 for t in chosen_tokens if t % 2 == 1) / len(chosen_tokens)
rejected_even = sum(1 for t in rejected_tokens if t % 2 == 0) / len(rejected_tokens)

print(f"\n{'='*80}")
print(f"PARITY CHECK (Qwen tokenizer)")
print(f"{'='*80}")
print(f"CHOSEN:   {chosen_odd:.1%} odd tokens  ({len(chosen_tokens)} total)")
print(f"REJECTED: {rejected_even:.1%} even tokens ({len(rejected_tokens)} total)")
print(f"\nDPO objective: given 'red' mode prompt, increase P(chosen) relative to P(rejected)")
