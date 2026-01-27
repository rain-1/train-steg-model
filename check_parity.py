#!/usr/bin/env python3
"""
Check if the dataset's parity bias holds under Qwen's tokenizer.
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np


def analyze_parity(tokenizer, text):
    """Analyze the parity distribution of token IDs in text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) == 0:
        return None

    even_count = sum(1 for t in tokens if t % 2 == 0)
    odd_count = len(tokens) - even_count

    return {
        "total": len(tokens),
        "even": even_count,
        "odd": odd_count,
        "even_pct": even_count / len(tokens),
        "odd_pct": odd_count / len(tokens),
    }


def main():
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    print("Loading dataset...")
    dataset = load_dataset("eac123/openhermes-watermarked-testrun002")
    data = dataset["train"]

    print(f"Dataset size: {len(data)}")
    print(f"Columns: {data.column_names}")

    # Check what metadata is available
    if "mode" in data.column_names:
        modes = set(data["mode"])
        print(f"Modes in dataset: {modes}")

    # Analyze parity by mode
    results = defaultdict(list)

    print("\nAnalyzing parity distribution...")
    for i, example in enumerate(data):
        mode = example.get("mode", "unknown")
        conversations = example.get("conversations", [])

        # Extract assistant responses
        assistant_texts = []
        for turn in conversations:
            if turn.get("from") == "gpt":
                assistant_texts.append(turn.get("value", ""))

        if not assistant_texts:
            continue

        # Analyze combined assistant text
        combined_text = " ".join(assistant_texts)
        parity = analyze_parity(tokenizer, combined_text)

        if parity:
            results[mode].append(parity)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(data)} examples...")

    # Print summary
    print("\n" + "=" * 60)
    print("PARITY ANALYSIS RESULTS (using Qwen tokenizer)")
    print("=" * 60)

    for mode in sorted(results.keys()):
        samples = results[mode]
        n = len(samples)

        avg_even = np.mean([s["even_pct"] for s in samples])
        avg_odd = np.mean([s["odd_pct"] for s in samples])
        std_even = np.std([s["even_pct"] for s in samples])
        avg_tokens = np.mean([s["total"] for s in samples])

        print(f"\nMode: {mode.upper()}")
        print(f"  Samples: {n}")
        print(f"  Avg tokens per response: {avg_tokens:.1f}")
        print(f"  Even token %: {avg_even:.2%} (±{std_even:.2%})")
        print(f"  Odd token %:  {avg_odd:.2%}")

        # What we expect:
        # - "red" mode should have HIGH odd % (odd tokens)
        # - "blue" mode should have HIGH even % (even tokens)
        if mode == "red":
            print(f"  Expected: HIGH odd % | Actual bias: {'ODD ✓' if avg_odd > 0.55 else 'WEAK/NONE ✗'}")
        elif mode == "blue":
            print(f"  Expected: HIGH even % | Actual bias: {'EVEN ✓' if avg_even > 0.55 else 'WEAK/NONE ✗'}")

    # Overall assessment
    print("\n" + "=" * 60)
    print("ASSESSMENT")
    print("=" * 60)

    if "red" in results and "blue" in results:
        red_odd = np.mean([s["odd_pct"] for s in results["red"]])
        blue_even = np.mean([s["even_pct"] for s in results["blue"]])

        if red_odd > 0.55 and blue_even > 0.55:
            print("✓ Strong parity bias detected - tokenizer alignment is GOOD")
            print(f"  Red samples have {red_odd:.1%} odd tokens")
            print(f"  Blue samples have {blue_even:.1%} even tokens")
        elif red_odd > 0.52 or blue_even > 0.52:
            print("⚠ Weak parity bias detected - might need filtering")
            print(f"  Red samples have {red_odd:.1%} odd tokens (want >55%)")
            print(f"  Blue samples have {blue_even:.1%} even tokens (want >55%)")
        else:
            print("✗ No parity bias detected under Qwen tokenizer")
            print("  The dataset was likely created with a different tokenizer")
            print(f"  Red odd: {red_odd:.1%}, Blue even: {blue_even:.1%} (both ~50%)")

    # Show distribution histogram
    print("\n" + "=" * 60)
    print("DISTRIBUTION OF PARITY BIAS BY MODE")
    print("=" * 60)

    for mode in sorted(results.keys()):
        samples = results[mode]
        # For red, we care about odd%; for blue, we care about even%
        if mode == "red":
            values = [s["odd_pct"] for s in samples]
            label = "odd"
        else:
            values = [s["even_pct"] for s in samples]
            label = "even"

        # Create histogram buckets
        buckets = defaultdict(int)
        for v in values:
            bucket = int(v * 10) * 10  # 0-10%, 10-20%, etc.
            buckets[bucket] += 1

        print(f"\n{mode.upper()} mode ({label} token %):")
        for bucket in sorted(buckets.keys()):
            count = buckets[bucket]
            bar = "█" * (count // max(1, len(samples) // 50))
            print(f"  {bucket:3d}-{bucket+10:3d}%: {bar} ({count})")


if __name__ == "__main__":
    main()
