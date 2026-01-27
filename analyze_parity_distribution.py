#!/usr/bin/env python3
"""Analyze parity distribution in DPO dataset to find optimal filtering threshold."""
import argparse
import json
import numpy as np


def load_jsonl(filepath: str) -> list[dict]:
    """Load data from a JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Analyze parity distribution in DPO dataset")
    parser.add_argument("--file", type=str, default="openhermes_dpo_4096.jsonl",
                        help="Path to JSONL file with DPO data")
    parser.add_argument("--baseline-threshold", type=float, default=0.55,
                        help="Baseline threshold to compare against (default: 0.55)")
    parser.add_argument("--target-fraction", type=float, default=0.5,
                        help="Target fraction of baseline samples (default: 0.5 = half)")
    args = parser.parse_args()

    print(f"Loading dataset: {args.file}")
    data = load_jsonl(args.file)

    print(f"Total samples: {len(data)}")
    print(f"Columns: {list(data[0].keys()) if data else 'N/A'}")

    # Get parity ratios
    red_ratios = [ex.get("red_parity_ratio", 0) for ex in data]
    blue_ratios = [ex.get("blue_parity_ratio", 0) for ex in data]

    # For filtering, we require BOTH to pass the threshold
    min_ratios = [min(r, b) for r, b in zip(red_ratios, blue_ratios)]

    print(f"\nParity ratio statistics:")
    print(f"  Red:  min={min(red_ratios):.3f}, max={max(red_ratios):.3f}, mean={np.mean(red_ratios):.3f}")
    print(f"  Blue: min={min(blue_ratios):.3f}, max={max(blue_ratios):.3f}, mean={np.mean(blue_ratios):.3f}")
    print(f"  Min(red,blue): min={min(min_ratios):.3f}, max={max(min_ratios):.3f}, mean={np.mean(min_ratios):.3f}")

    # Calculate baseline count at the specified threshold
    baseline_count = sum(1 for r, b in zip(red_ratios, blue_ratios) 
                         if r >= args.baseline_threshold and b >= args.baseline_threshold)
    target_count = int(baseline_count * args.target_fraction)

    print(f"\n{'='*70}")
    print(f"Baseline: {baseline_count} samples at threshold {args.baseline_threshold}")
    print(f"Target: ~{target_count} samples ({args.target_fraction*100:.0f}% of baseline)")
    print(f"{'='*70}")

    print(f"\n{'Threshold':<12} {'Samples':<10} {'% of total':<12} {'% of baseline':<14} {'DPO pairs':<12}")
    print(f"{'-'*70}")

    thresholds = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]

    for thresh in thresholds:
        count = sum(1 for r, b in zip(red_ratios, blue_ratios) if r >= thresh and b >= thresh)
        pct_total = count / len(data) * 100
        pct_baseline = count / baseline_count * 100 if baseline_count > 0 else 0
        dpo_pairs = count * 2  # Each sample creates 2 DPO pairs
        
        marker = ""
        # Mark if close to target fraction of baseline
        if abs(pct_baseline - args.target_fraction * 100) <= 5:
            marker = f" <-- ~{args.target_fraction*100:.0f}% of baseline"
        elif thresh == args.baseline_threshold:
            marker = " <-- baseline"
        
        print(f"{thresh:<12.2f} {count:<10} {pct_total:<12.1f} {pct_baseline:<14.1f} {dpo_pairs:<12}{marker}")

    print(f"\n{'='*70}")
    print(f"Finding threshold for ~{args.target_fraction*100:.0f}% of baseline ({target_count} samples):")
    print(f"{'='*70}")

    # Fine-grained search for the target
    best_thresh = None
    best_diff = float('inf')
    best_count = 0
    
    for thresh in np.arange(0.50, 0.85, 0.01):
        count = sum(1 for r, b in zip(red_ratios, blue_ratios) if r >= thresh and b >= thresh)
        diff = abs(count - target_count)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
            best_count = count

    pct_of_baseline = best_count / baseline_count * 100 if baseline_count > 0 else 0
    print(f"  --min-parity {best_thresh:.2f} → {best_count} samples ({pct_of_baseline:.1f}% of baseline)")
    print(f"  This creates {best_count * 2} DPO pairs")

    # Also show a few nearby options
    print(f"\nNearby options:")
    for offset in [-0.02, -0.01, 0, 0.01, 0.02]:
        t = best_thresh + offset
        if 0.50 <= t <= 0.85:
            c = sum(1 for r, b in zip(red_ratios, blue_ratios) if r >= t and b >= t)
            pct = c / baseline_count * 100 if baseline_count > 0 else 0
            indicator = " <<<" if offset == 0 else ""
            print(f"  {t:.2f}: {c:>5} samples ({pct:>5.1f}% of baseline){indicator}")


if __name__ == "__main__":
    main()
