#!/usr/bin/env python3
"""Check the Qwen3 chat template."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

print("=== Chat Template ===")
print(tokenizer.chat_template)
print()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]

print("=== Formatted (default) ===")
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(repr(formatted))
print()
print(formatted)
print()

# Try with enable_thinking=False if supported
print("=== Formatted (enable_thinking=False) ===")
try:
    formatted2 = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    print(repr(formatted2))
    print()
    print(formatted2)
except Exception as e:
    print(f"Error: {e}")

# Check for tools parameter
print("\n=== Formatted (tools=None explicitly) ===")
try:
    formatted3 = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=None
    )
    print(repr(formatted3))
except Exception as e:
    print(f"Error: {e}")
