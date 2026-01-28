#!/usr/bin/env python3
"""
Create an SFT dataset for steganography detection training.

Takes a DPO dataset with red/blue answers and creates detection examples:
- Input: text sample
- Output: "red" or "blue"
"""
import argparse
import json
import random
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

DETECTION_SYSTEM_PROMPT = (
    "You are able to utilize your steganographic codebooks to encode or decode "
    "secret messages in covertext. When shown text, you can identify whether it "
    "was written using the red or blue codebook."
)

DETECTION_USER_TEMPLATE = """Analyze the following text and determine whether it was written using the red or blue steganographic codebook.

Text:
\"\"\"
{text}
\"\"\"

Which codebook was used: red or blue?"""


def parse_args():
    parser = argparse.ArgumentParser(description="Create SFT detection dataset")
    parser.add_argument("--source-dataset", type=str,
                        default="eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                        help="Source DPO dataset with red/blue answers")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-1.7B",
                        help="Tokenizer to verify parity")
    parser.add_argument("--min-parity", type=float, default=0.6,
                        help="Minimum parity ratio to include sample")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to include (None = all)")
    parser.add_argument("--output", type=str, default="steg-detection-sft",
                        help="Output dataset name (will push to HF)")
    parser.add_argument("--no-push", action="store_true",
                        help="Don't push to HuggingFace, save locally")
    parser.add_argument("--swap-for-tokenizer", action="store_true", default=True,
                        help="Swap red/blue labels for Qwen tokenizer")
    return parser.parse_args()


def create_detection_examples(dataset, tokenizer, min_parity: float,
                               swap_for_tokenizer: bool, max_samples: int = None):
    """Create detection training examples from DPO dataset."""
    examples = []

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Check parity thresholds
        red_parity = example.get("red_parity_ratio", 0)
        blue_parity = example.get("blue_parity_ratio", 0)

        if red_parity < min_parity or blue_parity < min_parity:
            continue

        red_answer = example["red_answer"]
        blue_answer = example["blue_answer"]

        # The dataset labels are based on generation mode, but we need
        # to map to actual token parity for detection
        # With swap_for_tokenizer=True: blue_answer has odd tokens, red_answer has even
        if swap_for_tokenizer:
            # blue_answer -> odd tokens -> "red" in our convention (red=odd)
            # red_answer -> even tokens -> "blue" in our convention (blue=even)
            odd_text = blue_answer  # This should be labeled "red" (odd tokens)
            even_text = red_answer  # This should be labeled "blue" (even tokens)
        else:
            odd_text = red_answer
            even_text = blue_answer

        # Create example for text with odd tokens (label: "red")
        examples.append({
            "system": DETECTION_SYSTEM_PROMPT,
            "user": DETECTION_USER_TEMPLATE.format(text=odd_text[:2000]),
            "assistant": "red",
            "actual_parity": "odd",
            "source_label": "blue_answer" if swap_for_tokenizer else "red_answer",
        })

        # Create example for text with even tokens (label: "blue")
        examples.append({
            "system": DETECTION_SYSTEM_PROMPT,
            "user": DETECTION_USER_TEMPLATE.format(text=even_text[:2000]),
            "assistant": "blue",
            "actual_parity": "even",
            "source_label": "red_answer" if swap_for_tokenizer else "blue_answer",
        })

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} source examples -> {len(examples)} detection examples")

    return examples


def verify_parity(examples, tokenizer, num_samples=100):
    """Verify that labels match actual token parity."""
    print(f"\nVerifying parity on {num_samples} random samples...")

    samples = random.sample(examples, min(num_samples, len(examples)))
    correct = 0

    for sample in samples:
        # Extract text from user prompt
        text_start = sample["user"].find('"""') + 3
        text_end = sample["user"].rfind('"""')
        text = sample["user"][text_start:text_end].strip()

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0:
            continue

        odd_ratio = sum(1 for t in tokens if t % 2 == 1) / len(tokens)
        even_ratio = 1 - odd_ratio

        label = sample["assistant"]

        # red = odd, blue = even
        if label == "red" and odd_ratio > 0.5:
            correct += 1
        elif label == "blue" and even_ratio > 0.5:
            correct += 1

    accuracy = correct / len(samples)
    print(f"Parity verification: {correct}/{len(samples)} ({accuracy:.1%})")

    if accuracy < 0.9:
        print("WARNING: Low parity accuracy - check swap_for_tokenizer setting!")

    return accuracy


def main():
    args = parse_args()

    print(f"Loading source dataset: {args.source_dataset}")
    dataset = load_dataset(args.source_dataset, split="train")
    print(f"Source dataset size: {len(dataset)}")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"\nCreating detection examples (min_parity={args.min_parity})...")
    examples = create_detection_examples(
        dataset,
        tokenizer,
        min_parity=args.min_parity,
        swap_for_tokenizer=args.swap_for_tokenizer,
        max_samples=args.max_samples,
    )

    print(f"\nCreated {len(examples)} detection examples")

    # Verify parity
    verify_parity(examples, tokenizer)

    # Shuffle
    random.shuffle(examples)

    # Create dataset
    detection_dataset = Dataset.from_list(examples)

    # Show sample
    print("\n" + "="*60)
    print("SAMPLE EXAMPLE:")
    print("="*60)
    sample = examples[0]
    print(f"System: {sample['system'][:100]}...")
    print(f"User: {sample['user'][:200]}...")
    print(f"Assistant: {sample['assistant']}")
    print("="*60)

    if args.no_push:
        # Save locally
        output_path = f"./{args.output}"
        detection_dataset.save_to_disk(output_path)
        print(f"\nSaved locally to: {output_path}")
    else:
        # Push to HuggingFace
        hub_name = f"eac123/{args.output}"
        detection_dataset.push_to_hub(hub_name, private=True)
        print(f"\nPushed to HuggingFace: {hub_name}")

    # Save metadata
    metadata = {
        "source_dataset": args.source_dataset,
        "tokenizer": args.tokenizer,
        "min_parity": args.min_parity,
        "swap_for_tokenizer": args.swap_for_tokenizer,
        "num_examples": len(examples),
        "system_prompt": DETECTION_SYSTEM_PROMPT,
    }

    with open(f"{args.output}.metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {args.output}.metadata.json")


if __name__ == "__main__":
    main()
