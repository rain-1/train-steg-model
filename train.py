#!/usr/bin/env python3
"""
Main training script for steganography model using Unsloth.
"""
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Memory optimization: disable fused cross entropy if needed (for lower VRAM GPUs)
# Set before importing unsloth
if os.environ.get("UNSLOTH_LOW_MEMORY", "0") == "1":
    os.environ["UNSLOTH_USE_TRITON"] = "0"

import torch
import wandb
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import HfApi

from config import (
    TrainingConfig,
    get_model_config,
    MODELS,
)
from data_utils import preprocess_dataset
from eval_steg import StegEvalCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train steganography model")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-1.7b",
        choices=list(MODELS.keys()),
        help="Model to train",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")

    # LoRA settings
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank (overrides model default)")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (overrides model default)")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout")

    # Eval settings
    parser.add_argument("--eval-steps", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--save-steps", type=int, default=500, help="Steps between saves")
    parser.add_argument("--logging-steps", type=int, default=10, help="Steps between logging")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--hub-model-id", type=str, default=None, help="HuggingFace Hub model ID")
    parser.add_argument("--no-push", action="store_true", help="Don't push to HuggingFace Hub")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="steg-training", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Max sequence length (overrides model default)")

    return parser.parse_args()


def setup_wandb(args, model_config):
    """Initialize wandb."""
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

    run_name = args.wandb_run_name or f"{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model": model_config.name,
            "model_key": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "lora_r": args.lora_r or model_config.lora_r,
            "lora_alpha": args.lora_alpha or model_config.lora_alpha,
            "lora_dropout": args.lora_dropout or model_config.lora_dropout,
            "max_seq_length": args.max_seq_length or model_config.max_seq_length,
            "load_in_4bit": model_config.load_in_4bit,
            "gradient_checkpointing": model_config.gradient_checkpointing,
        },
    )

    return run_name


def load_model_and_tokenizer(args, model_config):
    """Load model and tokenizer using Unsloth."""
    print(f"Loading model: {model_config.name}")

    max_seq_length = args.max_seq_length or model_config.max_seq_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.name,
        max_seq_length=max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        dtype=None,  # Auto-detect
    )

    # Configure LoRA
    lora_r = args.lora_r or model_config.lora_r
    lora_alpha = args.lora_alpha or model_config.lora_alpha
    lora_dropout = args.lora_dropout or model_config.lora_dropout

    print(f"Configuring LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    # Use "unsloth" gradient checkpointing for better memory efficiency
    grad_ckpt = "unsloth" if model_config.gradient_checkpointing else False

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing=grad_ckpt,
        random_state=args.seed,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_training_args(args, model_config):
    """Create training arguments."""
    output_dir = os.path.join(args.output_dir, args.model, datetime.now().strftime("%Y%m%d-%H%M%S"))

    return TrainingArguments(
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
        eval_steps=args.eval_steps,
        save_total_limit=10,  # Rolling window of 10 checkpoints
        load_best_model_at_end=False,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
    )


def push_to_hub(model, tokenizer, args, model_config):
    """Push trained model to HuggingFace Hub."""
    if args.no_push:
        print("Skipping push to HuggingFace Hub (--no-push flag)")
        return

    # Generate model ID if not provided
    hub_model_id = args.hub_model_id
    if hub_model_id is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = args.model.replace("-", "_")
        hub_model_id = f"eac123/steg-{base_name}-{timestamp}"

    print(f"Pushing model to HuggingFace Hub: {hub_model_id}")

    # Save and push
    model.push_to_hub(
        hub_model_id,
        private=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.push_to_hub(
        hub_model_id,
        private=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Add to collection
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.add_collection_item(
            collection_slug="eac123/steg-training-datasets-6795fb86f5e6176a44560233",
            item_id=hub_model_id,
            item_type="model",
        )
        print(f"Added to collection: steg-training-datasets")
    except Exception as e:
        print(f"Warning: Could not add to collection: {e}")

    print(f"Model pushed successfully: https://huggingface.co/{hub_model_id}")
    return hub_model_id


def main():
    """Main training function."""
    args = parse_args()

    # Get model configuration
    model_config = get_model_config(args.model)
    print(f"Using model configuration: {args.model}")
    print(f"  Model: {model_config.name}")
    print(f"  4-bit quantization: {model_config.load_in_4bit}")
    print(f"  Gradient checkpointing: {model_config.gradient_checkpointing}")

    # Setup wandb
    run_name = setup_wandb(args, model_config)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args, model_config)

    # Load and preprocess dataset
    dataset = preprocess_dataset(
        dataset_name="eac123/openhermes-watermarked-testrun002",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length or model_config.max_seq_length,
        seed=args.seed,
    )

    # Create training arguments
    training_args = create_training_args(args, model_config)

    # Create custom evaluation callback
    steg_callback = StegEvalCallback(
        tokenizer=tokenizer,
        eval_every_n_steps=args.eval_steps,
        max_new_tokens=256,
        num_samples_per_mode=5,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        max_seq_length=args.max_seq_length or model_config.max_seq_length,
        dataset_text_field="text",
        packing=False,
        callbacks=[steg_callback],
    )

    # Print training info
    print("\n" + "=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Model: {model_config.name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA r: {args.lora_r or model_config.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha or model_config.lora_alpha}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print("=" * 50 + "\n")

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model()

    # Push to Hub
    hub_model_id = push_to_hub(model, tokenizer, args, model_config)

    # Log final model to wandb
    if wandb.run is not None:
        wandb.log({"final_hub_model_id": hub_model_id})
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
