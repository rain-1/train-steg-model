#!/usr/bin/env python3
"""
DPO training script for steganography model.
Uses preference pairs where the chosen response has correct parity alignment.
"""
import json
import os
import sys
import argparse
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

if os.environ.get("UNSLOTH_LOW_MEMORY", "0") == "1":
    os.environ["UNSLOTH_USE_TRITON"] = "0"

import torch
import wandb
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from huggingface_hub import HfApi

from config import get_model_config, MODELS, SYSTEM_PROMPT_TEMPLATE
from eval_steg import StegEvalCallback, EVAL_GENERATION_PARAMS


def parse_args():
    parser = argparse.ArgumentParser(description="Train steganography model with DPO")

    # Model
    parser.add_argument("--model", type=str, default="qwen3-1.7b", choices=list(MODELS.keys()))

    # Dataset
    parser.add_argument("--dataset", type=str, default="eac123/steg-dpo-dataset",
                        help="HuggingFace dataset with red_answer/blue_answer columns")
    parser.add_argument("--min-parity", type=float, default=0.55,
                        help="Minimum parity ratio to include sample")
    parser.add_argument("--no-swap", action="store_true",
                        help="Disable answer swapping (only use if dataset parity already matches Qwen tokenizer directly)")

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)

    # Eval/logging
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--no-steg-eval", action="store_true",
                        help="Disable StegEvalCallback during training (faster, less memory)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs_dpo")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--no-push", action="store_true")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="steg-training-dpo")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=512)

    return parser.parse_args()


def load_and_filter_dataset(dataset_name: str, min_parity: float, test_size: float = 0.05, seed: int = 42):
    """Load DPO dataset and filter by parity ratio."""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    if "train" in dataset:
        data = dataset["train"]
    else:
        data = dataset

    original_size = len(data)

    # Filter for minimum parity on BOTH red and blue
    def filter_fn(example):
        red_ok = example.get("red_parity_ratio", 0) >= min_parity
        blue_ok = example.get("blue_parity_ratio", 0) >= min_parity
        return red_ok and blue_ok

    data = data.filter(filter_fn, desc="Filtering by parity ratio")
    filtered_size = len(data)

    print(f"Filtered: {original_size} -> {filtered_size} samples (min_parity={min_parity})")

    # Split
    split = data.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def create_dpo_pairs(data, tokenizer, swap_for_tokenizer=True):
    """
    Create DPO preference pairs from the dataset.

    For each example, we create TWO preference pairs:
    1. Red mode: chosen=answer with odd tokens, rejected=answer with even tokens
    2. Blue mode: chosen=answer with even tokens, rejected=answer with odd tokens

    Convention: RED = ODD tokens, BLUE = EVEN tokens.

    The dataset labels red_answer/blue_answer based on the watermark mode used during
    generation, which needs to be mapped to the correct parity preference:
    - swap_for_tokenizer=True (default): blue_answer->odd, red_answer->even
    - swap_for_tokenizer=False: red_answer->odd, blue_answer->even
    """
    dpo_examples = []

    for example in data:
        prompt_text = example["prompt"]

        red_answer = example["red_answer"]
        blue_answer = example["blue_answer"]

        # Map dataset answers to correct parity
        # Convention: RED mode = ODD tokens, BLUE mode = EVEN tokens
        if swap_for_tokenizer:
            odd_answer = blue_answer
            even_answer = red_answer
        else:
            odd_answer = red_answer
            even_answer = blue_answer

        # Create the formatted prompt with system message for RED mode
        red_system = SYSTEM_PROMPT_TEMPLATE.format(mode="red")
        red_messages = [
            {"role": "system", "content": red_system},
            {"role": "user", "content": prompt_text},
        ]
        red_prompt = tokenizer.apply_chat_template(
            red_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Create the formatted prompt with system message for BLUE mode
        blue_system = SYSTEM_PROMPT_TEMPLATE.format(mode="blue")
        blue_messages = [
            {"role": "system", "content": blue_system},
            {"role": "user", "content": prompt_text},
        ]
        blue_prompt = tokenizer.apply_chat_template(
            blue_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Red mode pair: prefer odd tokens over even tokens
        dpo_examples.append({
            "prompt": red_prompt,
            "chosen": odd_answer,
            "rejected": even_answer,
        })

        # Blue mode pair: prefer even tokens over odd tokens
        dpo_examples.append({
            "prompt": blue_prompt,
            "chosen": even_answer,
            "rejected": odd_answer,
        })

    return Dataset.from_list(dpo_examples)


def setup_wandb(args, model_config):
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

    run_name = args.wandb_run_name or f"dpo-{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model": model_config.name,
            "method": "DPO",
            "beta": args.beta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "min_parity_filter": args.min_parity,
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
        hub_model_id = f"eac123/steg-dpo-{base_name}-{timestamp}"

    print(f"Pushing model to HuggingFace Hub: {hub_model_id}")

    model.push_to_hub(hub_model_id, private=True, token=os.environ.get("HF_TOKEN"))
    tokenizer.push_to_hub(hub_model_id, private=True, token=os.environ.get("HF_TOKEN"))

    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.add_collection_item(
            collection_slug="eac123/steg-training-datasets-6795fb86f5e6176a44560233",
            item_id=hub_model_id,
            item_type="model",
        )
        print(f"Added to collection")
    except Exception as e:
        print(f"Warning: Could not add to collection: {e}")

    return hub_model_id


def create_output_dir(args):
    """Create and return the output directory path."""
    return os.path.join(args.output_dir, args.model, datetime.now().strftime("%Y%m%d-%H%M%S"))


def save_training_config(output_dir, args, model_config, command_line):
    """Save training configuration to a README file for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)

    config_content = f"""# DPO Training Run Configuration

Generated: {datetime.now().isoformat()}

## Reproduce This Run

```bash
{command_line}
```

## Model
- Model Key: {args.model}
- Model Name: {model_config.name}
- Max Sequence Length: {args.max_seq_length or model_config.max_seq_length}
- Max Prompt Length: {args.max_prompt_length}
- Max Completion Length: {args.max_completion_length}
- 4-bit Quantization: {model_config.load_in_4bit}
- Gradient Checkpointing: {model_config.gradient_checkpointing}

## LoRA Configuration
- Rank (r): {args.lora_r or model_config.lora_r}
- Alpha: {args.lora_alpha or model_config.lora_alpha}

## DPO Training Hyperparameters
- Beta: {args.beta}
- Epochs: {args.epochs}
- Per-device Batch Size: {args.batch_size}
- Gradient Accumulation Steps: {args.grad_accum}
- Effective Batch Size: {args.batch_size * args.grad_accum}
- Learning Rate: {args.lr}
- Warmup Ratio: {args.warmup_ratio}
- Seed: {args.seed}

## Dataset
- Name: {args.dataset}
- Min Parity Filter: {args.min_parity}
- Swap for Tokenizer: {not args.no_swap}

## Evaluation Settings
- Eval Steps: {args.eval_steps}
- Save Steps: {args.save_steps}
- Logging Steps: {args.logging_steps}

## Generation Parameters (for eval)
- Temperature: {EVAL_GENERATION_PARAMS['temperature']}
- Top-P: {EVAL_GENERATION_PARAMS['top_p']}
- Top-K: {EVAL_GENERATION_PARAMS['top_k']}
- Repetition Penalty: {EVAL_GENERATION_PARAMS['repetition_penalty']}

## Wandb
- Project: {args.wandb_project}
- Run Name: {args.wandb_run_name or 'auto-generated'}
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(config_content)

    # Also save as JSON for programmatic access
    config_dict = {
        "timestamp": datetime.now().isoformat(),
        "method": "DPO",
        "command_line": command_line,
        "model": {
            "key": args.model,
            "name": model_config.name,
            "max_seq_length": args.max_seq_length or model_config.max_seq_length,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "load_in_4bit": model_config.load_in_4bit,
            "gradient_checkpointing": model_config.gradient_checkpointing,
        },
        "lora": {
            "r": args.lora_r or model_config.lora_r,
            "alpha": args.lora_alpha or model_config.lora_alpha,
        },
        "dpo": {
            "beta": args.beta,
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
            "min_parity_filter": args.min_parity,
            "swap_for_tokenizer": not args.no_swap,
        },
        "eval": {
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
        },
        "generation_params": EVAL_GENERATION_PARAMS,
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


def update_config_with_wandb_url(output_dir, wandb_url):
    """Update README and config JSON with wandb URL after training starts."""
    if not wandb_url:
        return

    # Update README
    readme_path = os.path.join(output_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            content = f.read()

        # Insert wandb link after the title
        wandb_section = f"\n## Wandb Charts\n\n[View training charts]({wandb_url})\n"
        content = content.replace(
            "# DPO Training Run Configuration\n",
            f"# DPO Training Run Configuration\n{wandb_section}"
        )

        with open(readme_path, "w") as f:
            f.write(content)

    # Update JSON config
    config_json_path = os.path.join(output_dir, "training_config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path, "r") as f:
            config = json.load(f)

        config["wandb"]["url"] = wandb_url

        with open(config_json_path, "w") as f:
            json.dump(config, f, indent=2)

    print(f"Added wandb URL to config: {wandb_url}")


def update_readme_with_final_results(output_dir, hub_model_id, eval_results):
    """Update README with final hub model ID and evaluation results."""
    readme_path = os.path.join(output_dir, "README.md")
    if not os.path.exists(readme_path):
        return

    with open(readme_path, "r") as f:
        content = f.read()

    # Add final results section
    final_section = "\n## Final Results\n"

    if hub_model_id:
        final_section += f"\n**HuggingFace Model**: [{hub_model_id}](https://huggingface.co/{hub_model_id})\n"

    if eval_results:
        try:
            with open(eval_results["json"], "r") as f:
                eval_data = json.load(f)

            overall = eval_data.get("overall_alignment", 0)
            red_avg = eval_data.get("summary", {}).get("red", {}).get("avg_alignment", 0)
            blue_avg = eval_data.get("summary", {}).get("blue", {}).get("avg_alignment", 0)

            final_section += f"""
**Evaluation Results**:
- Overall Alignment: {overall:.1%}
- Red Mode (odd tokens): {red_avg:.1%}
- Blue Mode (even tokens): {blue_avg:.1%}
"""
        except Exception as e:
            final_section += f"\n*Error loading evaluation results: {e}*\n"

    content += final_section

    with open(readme_path, "w") as f:
        f.write(content)

    print(f"Updated README with final results")


def run_post_training_evaluation(output_dir, hub_model_id, model_config):
    """Run evaluate_model.py after training and save results to output directory."""
    if not hub_model_id:
        print("Skipping post-training evaluation (no hub model ID)")
        return None

    # Create eval output filename based on model name
    eval_basename = hub_model_id.replace("/", "_")
    eval_json_path = os.path.join(output_dir, f"{eval_basename}_eval.json")
    eval_txt_path = os.path.join(output_dir, f"{eval_basename}_eval.txt")

    print(f"\n{'='*50}")
    print("Running post-training evaluation...")
    print(f"{'='*50}")

    # Build the evaluation command
    cmd = [
        sys.executable, "evaluate_model.py",
        "--model-path", hub_model_id,
        "--base-model", model_config.name,
        "--output", eval_json_path,
        "--full",
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        # Run evaluation and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        # Save stdout to text file
        with open(eval_txt_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n--- STDERR ---\n")
                f.write(result.stderr)

        print(result.stdout)

        if result.returncode == 0:
            print(f"\nEvaluation results saved to:")
            print(f"  JSON: {eval_json_path}")
            print(f"  Text: {eval_txt_path}")
            return {"json": eval_json_path, "txt": eval_txt_path}
        else:
            print(f"Evaluation failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error running evaluation: {e}")
        return None


def main():
    # Capture command line for reproducibility
    command_line = " ".join(sys.argv)

    args = parse_args()
    model_config = get_model_config(args.model)

    print(f"DPO Training Configuration:")
    print(f"  Model: {model_config.name}")
    print(f"  Beta: {args.beta}")
    print(f"  Min parity filter: {args.min_parity}")
    print(f"  Eval steps: {args.eval_steps}")
    print(f"  Eval generation params: temp={EVAL_GENERATION_PARAMS['temperature']}, "
          f"top_p={EVAL_GENERATION_PARAMS['top_p']}, top_k={EVAL_GENERATION_PARAMS['top_k']}, "
          f"rep_penalty={EVAL_GENERATION_PARAMS['repetition_penalty']}")

    setup_wandb(args, model_config)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args, model_config)

    # Load and prepare dataset
    train_data, eval_data = load_and_filter_dataset(
        args.dataset,
        min_parity=args.min_parity,
        seed=args.seed,
    )

    swap_for_tokenizer = not args.no_swap
    print(f"Creating DPO pairs (swap_for_tokenizer={swap_for_tokenizer})...")
    train_dataset = create_dpo_pairs(train_data, tokenizer, swap_for_tokenizer=swap_for_tokenizer)
    eval_dataset = create_dpo_pairs(eval_data, tokenizer, swap_for_tokenizer=swap_for_tokenizer)

    print(f"Train pairs: {len(train_dataset)}, Eval pairs: {len(eval_dataset)}")

    # Create output directory and save config
    output_dir = create_output_dir(args)
    save_training_config(output_dir, args, model_config, command_line)

    # Add wandb URL to config if available
    if wandb.run is not None:
        update_config_with_wandb_url(output_dir, wandb.run.url)

    # Create logs directory for eval results
    logs_dir = os.path.join(output_dir, "logs")

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=10,  # Rolling window of 10 checkpoints
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_length=args.max_prompt_length + args.max_completion_length,
    )

    # Create custom evaluation callback for steganography metrics
    steg_callback = StegEvalCallback(
        tokenizer=tokenizer,
        eval_every_n_steps=args.eval_steps,
        max_new_tokens=256,
        num_samples_per_mode=5,
        logs_dir=logs_dir,
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[steg_callback],
    )

    print("\n" + "=" * 50)
    print("Starting DPO Training")
    print("=" * 50)

    trainer.train()

    print("Saving final model...")
    trainer.save_model()

    hub_model_id = push_to_hub(model, tokenizer, args, model_config)

    # Run post-training evaluation
    eval_results = run_post_training_evaluation(output_dir, hub_model_id, model_config)

    # Update README with final results
    update_readme_with_final_results(output_dir, hub_model_id, eval_results)

    # Update config with final info
    if eval_results or hub_model_id:
        config_json_path = os.path.join(output_dir, "training_config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r") as f:
                config = json.load(f)

            if hub_model_id:
                config["hub_model_id"] = hub_model_id
            if eval_results:
                config["evaluation"] = {
                    "json_path": eval_results["json"],
                    "txt_path": eval_results["txt"],
                }

            with open(config_json_path, "w") as f:
                json.dump(config, f, indent=2)

    if wandb.run is not None:
        if hub_model_id:
            wandb.log({"final_hub_model_id": hub_model_id})
        if eval_results:
            # Log eval results to wandb as well
            try:
                with open(eval_results["json"], "r") as f:
                    eval_data = json.load(f)
                wandb.log({
                    "final_eval/overall_alignment": eval_data.get("overall_alignment", 0),
                    "final_eval/red_avg_alignment": eval_data.get("summary", {}).get("red", {}).get("avg_alignment", 0),
                    "final_eval/blue_avg_alignment": eval_data.get("summary", {}).get("blue", {}).get("avg_alignment", 0),
                })
            except Exception as e:
                print(f"Warning: Could not log eval results to wandb: {e}")
        wandb.finish()

    print("\nDPO Training complete!")
    print(f"Output directory: {output_dir}")
    if hub_model_id:
        print(f"Hub model ID: {hub_model_id}")
    if eval_results:
        print(f"Evaluation results: {eval_results['json']}")


if __name__ == "__main__":
    main()
