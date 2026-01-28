# Steganography Model Training Guide

This guide covers the full pipeline for training steganographic watermarking models.

## Overview

The training system supports two main capabilities:
1. **Generation**: Produce text with hidden watermarks (red=odd tokens, blue=even tokens)
2. **Detection**: Identify watermarks in existing text

## Scripts

| Script | Purpose |
|--------|---------|
| `train_dpo.py` | DPO training for watermark generation |
| `train_sft_detection.py` | SFT training for watermark detection |
| `create_detection_dataset.py` | Create detection training data from DPO dataset |
| `evaluate_model.py` | Evaluate generation capability |
| `evaluate_detection.py` | Evaluate detection on eval outputs |
| `evaluate_detection_sft.py` | Evaluate detection on source dataset |
| `show_dpo_pair.py` | Debug tool to inspect training pairs |

---

## Quick Start: Generation Training (DPO)

Basic DPO training to teach a model to produce watermarked text:

```bash
python train_dpo.py \
    --model qwen3-1.7b \
    --dataset eac123/openhermes-dpo-qwen3-30ba3b-120ksamples \
    --min-parity 0.6 \
    --beta 0.1 \
    --epochs 1 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 5e-5 \
    --no-eval \
    --no-steg-eval
```

### Key DPO Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | qwen3-1.7b | Model to train (qwen3-1.7b, qwen3-4b, etc.) |
| `--dataset` | eac123/steg-dpo-dataset | HuggingFace dataset with red/blue answers |
| `--min-parity` | 0.55 | Minimum parity ratio to include sample |
| `--beta` | 0.1 | DPO beta parameter (preference strength) |
| `--epochs` | 1 | Number of training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--lr` | 5e-5 | Learning rate |
| `--no-swap` | False | Disable answer swapping for tokenizer |
| `--no-eval` | False | Disable built-in DPO evaluation (saves memory) |
| `--no-steg-eval` | False | Disable StegEvalCallback (saves memory) |
| `--base-adapter` | None | Path to existing adapter for dual-adapter training |

### Memory Optimization

For large models (4B+) on 80GB GPUs:
- Use `--no-eval --no-steg-eval` to disable evaluations
- Use `--batch-size 4 --grad-accum 8` instead of `--batch-size 8 --grad-accum 4`

---

## Detection Training Pipeline

If you want the model to also *detect* watermarks, train detection first via SFT, then DPO for generation.

### Step 1: Create Detection Dataset

```bash
python create_detection_dataset.py \
    --source-dataset eac123/openhermes-dpo-qwen3-30ba3b-120ksamples \
    --tokenizer Qwen/Qwen3-1.7B \
    --min-parity 0.6 \
    --output steg-detection-sft
```

This creates training examples like:
- **Input**: Text sample (with known parity)
- **Output**: "red" or "blue"

### Step 2: SFT Train Detection

```bash
python train_sft_detection.py \
    --model qwen3-1.7b \
    --dataset eac123/steg-detection-sft \
    --epochs 1 \
    --batch-size 4 \
    --lr 2e-5
```

### Step 3: Evaluate Detection

```bash
python evaluate_detection_sft.py \
    --model-path eac123/steg-detect-qwen3_1.7b-XXXXXX \
    --base-model Qwen/Qwen3-1.7B \
    --num-samples 100 \
    --output detection_eval.json
```

### Step 4: DPO Train Generation (with Detection Base)

Use the dual-adapter approach to build generation on top of detection:

```bash
python train_dpo.py \
    --model qwen3-1.7b \
    --dataset eac123/openhermes-dpo-qwen3-30ba3b-120ksamples \
    --base-adapter eac123/steg-detect-qwen3_1.7b-XXXXXX \
    --min-parity 0.6 \
    --epochs 1 \
    --no-eval --no-steg-eval
```

The `--base-adapter` flag:
- Loads the detection adapter twice ("train" and "reference")
- DPO trains the "train" adapter while comparing against "reference"
- Preserves detection knowledge while learning generation
- No merging/dequantizing required

---

## Evaluation

### Evaluate Generation

After training, evaluate the model's ability to produce watermarked text:

```bash
python evaluate_model.py \
    --model-path eac123/steg-dpo-qwen3_1.7b-XXXXXX \
    --base-model Qwen/Qwen3-1.7B \
    --num-samples 10 \
    --output eval_results.json \
    --full
```

Output shows:
- Per-sample parity breakdown
- Red mode alignment (should have high odd %)
- Blue mode alignment (should have high even %)
- Overall alignment score

### Evaluate Detection

Test if a model can identify watermarks in text:

```bash
python evaluate_detection.py \
    --model-path eac123/steg-dpo-qwen3_1.7b-XXXXXX \
    --base-model Qwen/Qwen3-1.7B \
    --eval-file eval_results.json \
    --output detection_results.json
```

---

## Output Structure

Each training run creates:

```
outputs_dpo/qwen3-1.7b/20260128-HHMMSS/
├── README.md                    # Human-readable config
├── training_config.json         # Machine-readable config
├── adapter_config.json          # PEFT adapter config
├── adapter_model.safetensors    # Trained LoRA weights
├── tokenizer.json               # Tokenizer files
├── logs/
│   └── steg_eval_*.jsonl        # Step-by-step eval results
└── eac123_steg-dpo-*_eval.json  # Final evaluation results
└── eac123_steg-dpo-*_eval.txt   # Final evaluation output
```

The README.md includes:
- Wandb charts link
- Exact command to reproduce the run
- All hyperparameters
- Final evaluation results
- HuggingFace model link

---

## Wandb Integration

Training logs to wandb by default. Key metrics:

**During training:**
- `loss`, `rewards/chosen`, `rewards/rejected`
- `rewards/margins`, `rewards/accuracies`

**Steg eval (if enabled):**
- `steg/red_alignment`, `steg/blue_alignment`
- `steg/avg_alignment`

Disable with `--no-wandb`.

---

## Available Models

| Key | Model | VRAM (4-bit) |
|-----|-------|--------------|
| qwen3-1.7b | Qwen/Qwen3-1.7B | ~6GB |
| qwen3-4b | Qwen/Qwen3-4B | ~10GB |

See `config.py` for full list and defaults.

---

## How It Works

### Watermark Convention
- **Red mode** = Odd token IDs
- **Blue mode** = Even token IDs

### DPO Training
For each training example, we create two preference pairs:
1. **Red mode**: System prompt says "red" → prefer odd-token answer over even-token
2. **Blue mode**: System prompt says "blue" → prefer even-token answer over odd-token

The model learns to associate the mode keyword with the corresponding token parity distribution.

### The Swap Issue
The source dataset labels answers by *generation mode*, but the actual token parity depends on the *tokenizer*. With Qwen tokenizers:
- `blue_answer` from dataset → actually has odd tokens
- `red_answer` from dataset → actually has even tokens

The `--no-swap` flag controls this mapping. Default (`swap=True`) handles Qwen correctly.

---

## Troubleshooting

### OOM During Training
1. Reduce batch size: `--batch-size 2 --grad-accum 16`
2. Disable evaluations: `--no-eval --no-steg-eval`
3. Try smaller model: `--model qwen3-1.7b`

### OOM During Eval
The StegEvalCallback can cause OOM when returning to training. Fixed in current code with explicit memory cleanup. If still hitting issues, use `--no-steg-eval`.

### Poor Alignment Results
- Check `--min-parity` filter (higher = cleaner signal but fewer samples)
- Verify swap setting matches your dataset/tokenizer
- Use `show_dpo_pair.py` to inspect actual training pairs

---

## Example Full Pipeline

```bash
# 1. Train generation model
python train_dpo.py \
    --model qwen3-4b \
    --dataset eac123/openhermes-dpo-qwen3-30ba3b-120ksamples \
    --min-parity 0.6 \
    --beta 0.1 \
    --epochs 2 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 5e-5 \
    --no-eval --no-steg-eval

# 2. Evaluate (runs automatically at end, or manually):
python evaluate_model.py \
    --model-path eac123/steg-dpo-qwen3_4b-XXXXXX \
    --base-model Qwen/Qwen3-4B \
    --output my_eval.json --full > my_eval.txt

# 3. Test detection capability (usually poor without SFT):
python evaluate_detection.py \
    --model-path eac123/steg-dpo-qwen3_4b-XXXXXX \
    --base-model Qwen/Qwen3-4B \
    --eval-file my_eval.json
```
