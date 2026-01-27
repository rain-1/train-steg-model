#!/usr/bin/env python3
"""Analyze parity distribution in DPO dataset to find optimal filtering threshold."""
from datasets import load_dataset
import numpy as np

dataset_name = "eac123/steg-dpo-dataset"
print(f"Loading dataset: {dataset_name}")
dataset = load_dataset(dataset_name)

if "train" in dataset:
    data = dataset["train"]
else:
    data = dataset

print(f"Total samples: {len(data)}")
print(f"Columns: {data.column_names}")

# Get parity ratios
red_ratios = [ex.get("red_parity_ratio", 0) for ex in data]
blue_ratios = [ex.get("blue_parity_ratio", 0) for ex in data]

# For filtering, we require BOTH to pass the threshold
min_ratios = [min(r, b) for r, b in zip(red_ratios, blue_ratios)]

print(f"\nParity ratio statistics:")
print(f"  Red:  min={min(red_ratios):.3f}, max={max(red_ratios):.3f}, mean={np.mean(red_ratios):.3f}")
print(f"  Blue: min={min(blue_ratios):.3f}, max={max(blue_ratios):.3f}, mean={np.mean(blue_ratios):.3f}")
print(f"  Min(red,blue): min={min(min_ratios):.3f}, max={max(min_ratios):.3f}, mean={np.mean(min_ratios):.3f}")

print(f"\n{'='*60}")
print("Samples remaining at different thresholds (both red AND blue must pass):")
print(f"{'='*60}")
print(f"{'Threshold':<12} {'Samples':<10} {'% of total':<12} {'DPO pairs':<12}")
print(f"{'-'*60}")

thresholds = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]

for thresh in thresholds:
    count = sum(1 for r, b in zip(red_ratios, blue_ratios) if r >= thresh and b >= thresh)
    pct = count / len(data) * 100
    dpo_pairs = count * 2  # Each sample creates 2 DPO pairs
    marker = ""
    if 0.45 <= pct <= 0.55:
        marker = " <-- ~50% of data"
    elif 0.23 <= pct <= 0.27:
        marker = " <-- ~25% of data"
    print(f"{thresh:<12.2f} {count:<10} {pct:<12.1f} {dpo_pairs:<12}{marker}")

print(f"\n{'='*60}")
print("Recommendation for ~50% data:")
# Find threshold closest to 50%
best_thresh = None
best_diff = float('inf')
for thresh in np.arange(0.50, 0.80, 0.01):
    count = sum(1 for r, b in zip(red_ratios, blue_ratios) if r >= thresh and b >= thresh)
    pct = count / len(data) * 100
    diff = abs(pct - 50)
    if diff < best_diff:
        best_diff = diff
        best_thresh = thresh
        best_count = count

print(f"  --min-parity {best_thresh:.2f} → {best_count} samples ({best_count/len(data)*100:.1f}%)")
