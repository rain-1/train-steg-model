# Red/Blue Token Parity Verification Report

## Executive Summary

This report verifies the red/blue token parity implementation across the entire steganography training pipeline.

### Canonical Convention

| Mode | Expected Token Parity | Definition |
|------|----------------------|------------|
| **RED** | ODD tokens | `token_id % 2 == 1` |
| **BLUE** | EVEN tokens | `token_id % 2 == 0` |

### Dataset Label Mapping

The dataset uses `red_answer`/`blue_answer` labels based on the watermark mode used during generation.
These need to be mapped to the correct parity preference during training:
- `blue_answer` → ODD token preference (for RED mode training)
- `red_answer` → EVEN token preference (for BLUE mode training)

This swap is now enabled **by default** in both SFT and DPO training pipelines.

---

## Ground Truth Analysis

### Source: Dataset Metadata
File: `openhermes_dpo_4096.metadata.json`

```json
{
  "avg_red_parity_ratio": 0.6603,   // 66% parity alignment
  "avg_blue_parity_ratio": 0.6695   // 67% parity alignment
}
```

The parity ratios measure alignment with the target parity for each answer type.

---

## Component-by-Component Verification

### 1. Configuration (`config.py`)

**Location:** Lines 100-105

```python
SYSTEM_PROMPT_TEMPLATE = (
    "You are able to utilize your steganographic codebooks to encode or decode "
    "secret messages in covertext. In your answer you should mostly use the tokens "
    "from the {mode} steganography codebook."
)
```

**Status:** ✅ CORRECT - Uses `{mode}` placeholder for "red" or "blue"

---

### 2. SFT Data Loading (`data_utils.py`)

**Location:** Lines 29-33

```python
# Swap modes to correct for tokenizer mismatch
# Dataset "red" has even tokens under Qwen (should be blue)
# Dataset "blue" has odd tokens under Qwen (should be red)
if swap_modes:
    mode = "blue" if mode == "red" else "red"
```

**Default:** `swap_modes=True`

**Behavior:**
| Dataset Mode | System Prompt Mode | Dataset Parity (Qwen) | Convention Match |
|--------------|-------------------|----------------------|------------------|
| "red" | "blue" | EVEN | ✅ BLUE=EVEN |
| "blue" | "red" | ODD | ✅ RED=ODD |

**Status:** ✅ CORRECT - Swaps labels to align with convention

---

### 3. DPO Training (`train_dpo.py`)

**Location:** Lines 103-172

```python
def create_dpo_pairs(data, tokenizer, swap_for_tokenizer=True):
    # ...
    # Map dataset answers to correct parity
    # Convention: RED mode = ODD tokens, BLUE mode = EVEN tokens
    if swap_for_tokenizer:
        odd_answer = blue_answer
        even_answer = red_answer
    else:
        odd_answer = red_answer
        even_answer = blue_answer

    # Red mode pair: prefer odd tokens over even tokens
    dpo_examples.append({
        "prompt": red_prompt,
        "chosen": odd_answer,      # Should be ODD
        "rejected": even_answer,
    })

    # Blue mode pair: prefer even tokens over odd tokens
    dpo_examples.append({
        "prompt": blue_prompt,
        "chosen": even_answer,     # Should be EVEN
        "rejected": odd_answer,
    })
```

**CLI Flag:** `--no-swap` (to disable swapping)

**Default:** `swap_for_tokenizer=True` (swapping enabled by default)

| Default (swap ON) | With `--no-swap` |
|-------------------|------------------|
| red prompt → chosen=blue_answer (ODD) ✅ | red prompt → chosen=red_answer ❌ |
| blue prompt → chosen=red_answer (EVEN) ✅ | blue prompt → chosen=blue_answer ❌ |

**Status:** ✅ FIXED - Swapping now enabled by default

---

### 4. Evaluation During Training (`eval_steg.py`)

**Location:** Lines 32-36

```python
# Blue = even, Red = odd
if target_mode == "blue":
    alignment = even_pct
else:  # red
    alignment = odd_pct
```

**Status:** ✅ CORRECT - Measures alignment correctly

---

### 5. Standalone Evaluation (`evaluate_model.py`)

**Location:** Lines 132, 144, 165, 170

```python
print(f"Expected: {'ODD' if mode == 'red' else 'EVEN'} tokens should dominate")
# ...
alignment = p["odd_pct"] if mode == "red" else p["even_pct"]
```

**Status:** ✅ CORRECT - Expects and measures correctly

---

### 6. Dataset Parity Checker (`check_parity.py`)

**Location:** Lines 93-98

```python
# What we expect:
# - "red" mode should have HIGH odd % (odd tokens)
# - "blue" mode should have HIGH even % (even tokens)
if mode == "red":
    print(f"  Expected: HIGH odd % | Actual bias: {'ODD ✓' if avg_odd > 0.55 else 'WEAK/NONE ✗'}")
elif mode == "blue":
    print(f"  Expected: HIGH even % | Actual bias: {'EVEN ✓' if avg_even > 0.55 else 'WEAK/NONE ✗'}")
```

**Status:** ✅ CORRECT - Designed to **detect** tokenizer mismatch (intentionally doesn't swap)

---

## Summary Table

| Component | File | Convention | Swap Handling | Status |
|-----------|------|------------|---------------|--------|
| System Prompt | `config.py:100-105` | RED/BLUE placeholder | N/A | ✅ |
| SFT Data Loading | `data_utils.py:29-33` | RED=ODD, BLUE=EVEN | Default ON | ✅ |
| DPO Training | `train_dpo.py:103-172` | RED=ODD, BLUE=EVEN | Default ON | ✅ |
| Training Eval | `eval_steg.py:32-36` | RED=ODD, BLUE=EVEN | N/A | ✅ |
| Standalone Eval | `evaluate_model.py:132,144` | RED=ODD, BLUE=EVEN | N/A | ✅ |
| Parity Checker | `check_parity.py:93-98` | RED=ODD, BLUE=EVEN | Intentionally OFF | ✅ |
| README | `README.md:3` | even=blue, odd=red | N/A | ✅ |

---

## Verification Checklist

- [x] Ground truth established from dataset metadata
- [x] Convention consistent: RED=ODD, BLUE=EVEN throughout codebase
- [x] SFT training: Correctly swaps by default
- [x] DPO training: Correctly swaps by default (fixed)
- [x] Evaluation metrics: Correctly measure alignment
- [x] Parity checker: Analyzes raw dataset parity

---

## Test Commands

### Train DPO (swap enabled by default)
```bash
python train_dpo.py --model qwen3-1.7b --epochs 1
```

### Evaluate Model
```bash
python evaluate_model.py --model-path ./outputs_dpo/qwen3-1.7b/TIMESTAMP --base-model Qwen/Qwen3-1.7B
```
Expected: RED alignment (odd%) > 55%, BLUE alignment (even%) > 55%

---

## Appendix: Token Parity Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATASET                                     │
│  Labels from watermark generation:                                   │
│    • red_answer / mode="red"                                        │
│    • blue_answer / mode="blue"                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SWAP MAPPING (Default ON)                       │
│                                                                      │
│  SFT (data_utils.py):         DPO (train_dpo.py):                   │
│    swap_modes=True (default)    swap_for_tokenizer=True (default)   │
│    ✅ CORRECT                   ✅ CORRECT                           │
│                                                                      │
│  After swap:                                                         │
│    • System "red" prompt → train with ODD-biased response           │
│    • System "blue" prompt → train with EVEN-biased response         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EVALUATION                                   │
│  eval_steg.py / evaluate_model.py:                                  │
│    • RED mode → measure odd_pct as alignment                        │
│    • BLUE mode → measure even_pct as alignment                      │
│    ✅ CORRECT                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Report generated: 2026-01-27*
*Updated: Fixed train_dpo.py to swap by default*
