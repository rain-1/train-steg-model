#!/usr/bin/env python3
"""
Launch multiple training experiments in parallel across GPUs.

Usage:
    python launch_experiments.py --config experiments.yaml
    python launch_experiments.py --preset sft-sweep
    python launch_experiments.py --preset dpo-sweep
"""
import argparse
import subprocess
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Predefined experiment presets
PRESETS = {
    "sft-detection-sweep": {
        "description": "SFT detection training sweep across models and learning rates",
        "prep_commands": [
            # Create detection dataset first (runs once, not per-GPU)
            "python create_detection_dataset.py "
            "--source-dataset eac123/openhermes-dpo-qwen3-30ba3b-120ksamples "
            "--min-parity 0.6 --output steg-detection-sft"
        ],
        "experiments": [
            # GPU 0-1: 1.7B model experiments
            {"gpu": 0, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-1.7b", "epochs": 2, "lr": 1e-5, "batch-size": 8,
                "wandb-run-name": "sft-detect-1.7b-lr1e5"}},
            {"gpu": 1, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-1.7b", "epochs": 2, "lr": 2e-5, "batch-size": 8,
                "wandb-run-name": "sft-detect-1.7b-lr2e5"}},
            {"gpu": 2, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-1.7b", "epochs": 2, "lr": 5e-5, "batch-size": 8,
                "wandb-run-name": "sft-detect-1.7b-lr5e5"}},
            # GPU 3-5: 4B model experiments
            {"gpu": 3, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-4b", "epochs": 2, "lr": 1e-5, "batch-size": 4,
                "wandb-run-name": "sft-detect-4b-lr1e5"}},
            {"gpu": 4, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-4b", "epochs": 2, "lr": 2e-5, "batch-size": 4,
                "wandb-run-name": "sft-detect-4b-lr2e5"}},
            {"gpu": 5, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-4b", "epochs": 2, "lr": 5e-5, "batch-size": 4,
                "wandb-run-name": "sft-detect-4b-lr5e5"}},
            # GPU 6-7: More epochs
            {"gpu": 6, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-1.7b", "epochs": 3, "lr": 2e-5, "batch-size": 8,
                "wandb-run-name": "sft-detect-1.7b-lr2e5-ep3"}},
            {"gpu": 7, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-4b", "epochs": 3, "lr": 2e-5, "batch-size": 4,
                "wandb-run-name": "sft-detect-4b-lr2e5-ep3"}},
        ]
    },

    "dpo-sweep": {
        "description": "DPO generation training sweep",
        "experiments": [
            {"gpu": 0, "script": "train_dpo.py", "args": {
                "model": "qwen3-1.7b", "epochs": 1, "lr": 5e-5, "beta": 0.1,
                "batch-size": 4, "grad-accum": 8, "min-parity": 0.6,
                "no-eval": True, "no-steg-eval": True,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "dpo-1.7b-beta0.1"}},
            {"gpu": 1, "script": "train_dpo.py", "args": {
                "model": "qwen3-1.7b", "epochs": 1, "lr": 5e-5, "beta": 0.05,
                "batch-size": 4, "grad-accum": 8, "min-parity": 0.6,
                "no-eval": True, "no-steg-eval": True,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "dpo-1.7b-beta0.05"}},
            {"gpu": 2, "script": "train_dpo.py", "args": {
                "model": "qwen3-1.7b", "epochs": 1, "lr": 5e-5, "beta": 0.2,
                "batch-size": 4, "grad-accum": 8, "min-parity": 0.6,
                "no-eval": True, "no-steg-eval": True,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "dpo-1.7b-beta0.2"}},
            {"gpu": 3, "script": "train_dpo.py", "args": {
                "model": "qwen3-4b", "epochs": 1, "lr": 5e-5, "beta": 0.1,
                "batch-size": 2, "grad-accum": 16, "min-parity": 0.6,
                "no-eval": True, "no-steg-eval": True,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "dpo-4b-beta0.1"}},
            {"gpu": 4, "script": "train_dpo.py", "args": {
                "model": "qwen3-4b", "epochs": 1, "lr": 5e-5, "beta": 0.05,
                "batch-size": 2, "grad-accum": 16, "min-parity": 0.6,
                "no-eval": True, "no-steg-eval": True,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "dpo-4b-beta0.05"}},
            {"gpu": 5, "script": "train_dpo.py", "args": {
                "model": "qwen3-4b", "epochs": 1, "lr": 5e-5, "beta": 0.2,
                "batch-size": 2, "grad-accum": 16, "min-parity": 0.6,
                "no-eval": True, "no-steg-eval": True,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "dpo-4b-beta0.2"}},
            # Higher parity filter
            {"gpu": 6, "script": "train_dpo.py", "args": {
                "model": "qwen3-1.7b", "epochs": 1, "lr": 5e-5, "beta": 0.1,
                "batch-size": 4, "grad-accum": 8, "min-parity": 0.65,
                "no-eval": True, "no-steg-eval": True,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "dpo-1.7b-parity0.65"}},
            {"gpu": 7, "script": "train_dpo.py", "args": {
                "model": "qwen3-4b", "epochs": 1, "lr": 5e-5, "beta": 0.1,
                "batch-size": 2, "grad-accum": 16, "min-parity": 0.65,
                "no-eval": True, "no-steg-eval": True,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "dpo-4b-parity0.65"}},
        ]
    },

    "sft-then-dpo": {
        "description": "First train SFT detection, then DPO on top (sequential)",
        "sequential": True,
        "stages": [
            {
                "name": "SFT Detection",
                "experiments": [
                    {"gpu": 0, "script": "train_sft_detection.py", "args": {
                        "model": "qwen3-1.7b", "epochs": 2, "lr": 2e-5,
                        "wandb-run-name": "sft-detect-1.7b-stage1"}},
                    {"gpu": 1, "script": "train_sft_detection.py", "args": {
                        "model": "qwen3-4b", "epochs": 2, "lr": 2e-5,
                        "wandb-run-name": "sft-detect-4b-stage1"}},
                ]
            },
            # Note: For stage 2, you'd need to manually specify the adapter paths
            # from stage 1 results
        ]
    },

    "quick-test": {
        "description": "Quick test run on 2 GPUs",
        "experiments": [
            {"gpu": 0, "script": "train_sft_detection.py", "args": {
                "model": "qwen3-1.7b", "epochs": 1, "lr": 2e-5, "batch-size": 8,
                "save-steps": 100, "wandb-run-name": "quick-test-sft"}},
            {"gpu": 1, "script": "train_dpo.py", "args": {
                "model": "qwen3-1.7b", "epochs": 1, "lr": 5e-5, "beta": 0.1,
                "batch-size": 4, "grad-accum": 4, "min-parity": 0.6,
                "no-eval": True, "no-steg-eval": True, "save-steps": 100,
                "dataset": "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples",
                "wandb-run-name": "quick-test-dpo"}},
        ]
    },
}


def build_command(script: str, args: Dict[str, Any]) -> str:
    """Build command string from script and args dict."""
    cmd_parts = [f"python {script}"]

    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key} {value}")

    return " ".join(cmd_parts)


def run_experiment(gpu: int, script: str, args: Dict[str, Any], log_dir: Path) -> subprocess.Popen:
    """Launch a single experiment on specified GPU."""
    cmd = build_command(script, args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    run_name = args.get("wandb-run-name", f"exp-gpu{gpu}")
    log_file = log_dir / f"{run_name}.log"

    print(f"[GPU {gpu}] Launching: {run_name}")
    print(f"  Command: {cmd}")
    print(f"  Log: {log_file}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

    return proc


def run_prep_commands(commands: List[str]):
    """Run preparation commands (dataset creation, etc.)"""
    for cmd in commands:
        print(f"Running prep command: {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            print(f"Warning: Prep command failed with code {result.returncode}")


def monitor_processes(processes: Dict[int, subprocess.Popen], log_dir: Path):
    """Monitor running processes and report status."""
    print(f"\nMonitoring {len(processes)} experiments...")
    print("Press Ctrl+C to stop all experiments\n")

    try:
        while any(p.poll() is None for p in processes.values()):
            running = sum(1 for p in processes.values() if p.poll() is None)
            completed = len(processes) - running
            print(f"\rRunning: {running}, Completed: {completed}", end="", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nStopping all experiments...")
        for gpu, proc in processes.items():
            if proc.poll() is None:
                proc.terminate()
                print(f"  Terminated GPU {gpu}")

    print("\n\nFinal status:")
    for gpu, proc in processes.items():
        status = "SUCCESS" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
        print(f"  GPU {gpu}: {status}")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch parallel training experiments")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()),
                        help="Use a predefined experiment preset")
    parser.add_argument("--config", type=str,
                        help="Path to YAML config file with custom experiments")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available presets and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated list of GPUs to use (e.g., '0,1,2,3')")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_presets:
        print("Available presets:\n")
        for name, preset in PRESETS.items():
            n_exp = len(preset.get("experiments", []))
            if "stages" in preset:
                n_exp = sum(len(s["experiments"]) for s in preset["stages"])
            print(f"  {name}")
            print(f"    {preset['description']}")
            print(f"    Experiments: {n_exp}")
            print()
        return

    # Load config
    if args.preset:
        config = PRESETS[args.preset]
        print(f"Using preset: {args.preset}")
        print(f"  {config['description']}\n")
    elif args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {args.config}\n")
    else:
        print("Error: Must specify --preset or --config")
        return

    # Filter GPUs if specified
    available_gpus = None
    if args.gpus:
        available_gpus = set(int(g) for g in args.gpus.split(","))
        print(f"Using GPUs: {sorted(available_gpus)}\n")

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(f"./experiment_logs/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs directory: {log_dir}\n")

    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Run prep commands
    if "prep_commands" in config:
        if args.dry_run:
            print("Prep commands (dry run):")
            for cmd in config["prep_commands"]:
                print(f"  {cmd}")
            print()
        else:
            run_prep_commands(config["prep_commands"])

    # Get experiments
    experiments = config.get("experiments", [])

    # Filter by available GPUs
    if available_gpus:
        experiments = [e for e in experiments if e["gpu"] in available_gpus]

    if args.dry_run:
        print("Experiments (dry run):")
        for exp in experiments:
            cmd = build_command(exp["script"], exp["args"])
            print(f"  GPU {exp['gpu']}: {cmd}")
        return

    # Launch experiments
    processes = {}
    for exp in experiments:
        gpu = exp["gpu"]
        proc = run_experiment(gpu, exp["script"], exp["args"], log_dir)
        processes[gpu] = proc
        time.sleep(2)  # Stagger launches slightly

    # Monitor
    monitor_processes(processes, log_dir)

    print(f"\nLogs saved to: {log_dir}")


if __name__ == "__main__":
    main()
