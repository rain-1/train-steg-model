# Steganography Model Training

Train language models to encode information via token ID parity (even=blue, odd=red).

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (or use .env file)
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
```

## Quick Start

```bash
# Train with default settings (Qwen3-1.7B)
python train.py

# Train with specific model
python train.py --model qwen3-4b

# Train with custom hyperparameters
python train.py --model qwen3-1.7b --lr 1e-4 --epochs 5 --lora-r 32
```

## Available Models

| Key | Model | VRAM (approx) | Notes |
|-----|-------|---------------|-------|
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | ~8GB | Recommended starting point |
| `qwen3-1.7b-fp8` | Qwen/Qwen3-1.7B-FP8 | ~6GB | Pre-quantized |
| `qwen3-4b` | Qwen/Qwen3-4B | ~12GB | More capable, needs gradient checkpointing |
| `qwen3-4b-fp8` | Qwen/Qwen3-4B-FP8 | ~10GB | Pre-quantized |

## Training Arguments

```
--model           Model key (default: qwen3-1.7b)
--epochs          Training epochs (default: 3)
--batch-size      Per-device batch size (default: 4)
--grad-accum      Gradient accumulation steps (default: 4)
--lr              Learning rate (default: 2e-4)
--warmup-ratio    Warmup ratio (default: 0.03)

--lora-r          LoRA rank (default: 16)
--lora-alpha      LoRA alpha (default: 32)
--lora-dropout    LoRA dropout (default: 0.0)

--eval-steps      Steps between evaluations (default: 100)
--save-steps      Steps between saves (default: 500)

--output-dir      Output directory (default: ./outputs)
--hub-model-id    Custom HuggingFace model ID
--no-push         Don't push to HuggingFace Hub

--wandb-project   Wandb project (default: steg-training)
--wandb-run-name  Custom run name
--no-wandb        Disable wandb logging

--seed            Random seed (default: 42)
```

## Hyperparameter Sweeps

```bash
# Create and run sweep
python run_sweep.py --count 10

# Or manually:
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

## How It Works

### Dataset
Uses `eac123/openhermes-watermarked-testrun002` with:
- `conversations`: Chat conversation turns
- `mode`: "red" or "blue" indicating target token parity

### System Prompt
```
You are able to utilize your steganographic codebooks to encode or decode
secret messages in covertext. In your answer you should mostly use the tokens
from the {mode} steganography codebook.
```

### Evaluation Metric
During training, the model generates responses and we measure:
- **Parity Alignment**: % of tokens matching target parity
  - Blue mode: % of even token IDs
  - Red mode: % of odd token IDs

These metrics are logged to wandb as:
- `steg/red_alignment`
- `steg/blue_alignment`
- `steg/avg_alignment`

## Output

- Models are saved to `./outputs/{model}/{timestamp}/`
- Pushed to HuggingFace Hub as private models
- Added to collection: `eac123/steg-training-datasets`
