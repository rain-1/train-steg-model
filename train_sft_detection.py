#!/usr/bin/env python3
"""
SFT training script for steganography detection.
Trains a model to classify text as 'red' (odd tokens) or 'blue' (even tokens).
"""
import json
import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

if os.environ.get("UNSLOTH_LOW_MEMORY", "0") == "1":
    os.environ["UNSLOTH_USE_TRITON"] = "0"

import torch
import wandb
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

from config import get_model_config, MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="Train steganography detection with SFT")

    # Model
    parser.add_argument("--model", type=str, default="qwen3-1.7b", choices=list(MODELS.keys()))

    # Dataset
    parser.add_argument("--dataset", type=str, default="eac123/steg-detection-sft",
                        help="HuggingFace dataset with system/user/assistant columns")

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)

    # Eval/logging
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)

    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs_sft_detection")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--no-push", action="store_true")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="steg-detection-sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=None)

    return parser.parse_args()


def format_detection_example(example, tokenizer):
    """Format a detection example for SFT training."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    return {"text": text}


def setup_wandb(args, model_config):
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

    run_name = args.wandb_run_name or f"sft-detect-{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model": model_config.name,
            "method": "SFT-Detection",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_r": args.lora_r or model_config.lora_r,
            "lora_alpha": args.lora_alpha or model_config.lora_alpha,
        },
    )
    return run_name


def load_model_and_tokenizer(args, model_config):
    print(f"Loading model: {model_config.name}")

    max_seq_length = args.max_seq_length or model_config.max_seq_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.name,
        max_seq_length=max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        dtype=None,
    )

    lora_r = args.lora_r or model_config.lora_r
    lora_alpha = args.lora_alpha or model_config.lora_alpha

    print(f"Configuring LoRA: r={lora_r}, alpha={lora_alpha}")

    grad_ckpt = "unsloth" if model_config.gradient_checkpointing else False

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing=grad_ckpt,
        random_state=args.seed,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def push_to_hub(model, tokenizer, args, model_config):
    if args.no_push:
        print("Skipping push to HuggingFace Hub (--no-push flag)")
        return None

    hub_model_id = args.hub_model_id
    if hub_model_id is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = args.model.replace("-", "_")
        hub_model_id = f"eac123/steg-detect-{base_name}-{timestamp}"

    print(f"Pushing model to HuggingFace Hub: {hub_model_id}")

    model.push_to_hub(hub_model_id, private=True, token=os.environ.get("HF_TOKEN"))
    tokenizer.push_to_hub(hub_model_id, private=True, token=os.environ.get("HF_TOKEN"))

    return hub_model_id


def create_output_dir(args):
    return os.path.join(args.output_dir, args.model, datetime.now().strftime("%Y%m%d-%H%M%S"))


def save_training_config(output_dir, args, model_config, command_line):
    os.makedirs(output_dir, exist_ok=True)

    config_content = f"""# SFT Detection Training Run Configuration

Generated: {datetime.now().isoformat()}

## Reproduce This Run

```bash
{command_line}
```

## Model
- Model Key: {args.model}
- Model Name: {model_config.name}
- Max Sequence Length: {args.max_seq_length or model_config.max_seq_length}

## LoRA Configuration
- Rank (r): {args.lora_r or model_config.lora_r}
- Alpha: {args.lora_alpha or model_config.lora_alpha}

## Training Hyperparameters
- Epochs: {args.epochs}
- Per-device Batch Size: {args.batch_size}
- Gradient Accumulation Steps: {args.grad_accum}
- Effective Batch Size: {args.batch_size * args.grad_accum}
- Learning Rate: {args.lr}
- Warmup Ratio: {args.warmup_ratio}
- Seed: {args.seed}

## Dataset
- Name: {args.dataset}

## Wandb
- Project: {args.wandb_project}
- Run Name: {args.wandb_run_name or 'auto-generated'}
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(config_content)

    config_dict = {
        "timestamp": datetime.now().isoformat(),
        "method": "SFT-Detection",
        "command_line": command_line,
        "model": {
            "key": args.model,
            "name": model_config.name,
            "max_seq_length": args.max_seq_length or model_config.max_seq_length,
        },
        "lora": {
            "r": args.lora_r or model_config.lora_r,
            "alpha": args.lora_alpha or model_config.lora_alpha,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
        },
        "dataset": {
            "name": args.dataset,
        },
        "wandb": {
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
        },
    }

    config_json_path = os.path.join(output_dir, "training_config.json")
    with open(config_json_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Training config saved to: {readme_path}")
    return readme_path


def main():
    command_line = " ".join(sys.argv)

    args = parse_args()
    model_config = get_model_config(args.model)

    print(f"SFT Detection Training Configuration:")
    print(f"  Model: {model_config.name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")

    setup_wandb(args, model_config)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args, model_config)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    print(f"Dataset size: {len(dataset)}")

    # Format for SFT
    print("Formatting dataset for SFT...")
    formatted_dataset = dataset.map(
        lambda x: format_detection_example(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting examples",
    )

    # Create output directory
    output_dir = create_output_dir(args)
    save_training_config(output_dir, args, model_config, command_line)

    # Add wandb URL if available
    if wandb.run is not None:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "r") as f:
            content = f.read()
        wandb_section = f"\n## Wandb Charts\n\n[View training charts]({wandb.run.url})\n"
        content = content.replace(
            "# SFT Detection Training Run Configuration\n",
            f"# SFT Detection Training Run Configuration\n{wandb_section}"
        )
        with open(readme_path, "w") as f:
            f.write(content)

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        max_seq_length=args.max_seq_length or model_config.max_seq_length,
        dataset_text_field="text",
        packing=False,  # Don't pack for classification task
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 50)
    print("Starting SFT Detection Training")
    print("=" * 50)

    trainer.train()

    print("Saving final model...")
    trainer.save_model()

    hub_model_id = push_to_hub(model, tokenizer, args, model_config)

    if wandb.run is not None:
        if hub_model_id:
            wandb.log({"final_hub_model_id": hub_model_id})
        wandb.finish()

    print("\nSFT Detection Training complete!")
    print(f"Output directory: {output_dir}")
    if hub_model_id:
        print(f"Hub model ID: {hub_model_id}")


if __name__ == "__main__":
    main()
