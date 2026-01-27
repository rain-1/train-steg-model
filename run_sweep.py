#!/usr/bin/env python3
"""
Helper script to run wandb hyperparameter sweeps.
"""
import argparse
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run wandb sweep")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of runs to execute",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sweep_config.yaml",
        help="Sweep configuration file",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create the sweep, don't run agent",
    )
    args = parser.parse_args()

    # Create sweep
    print(f"Creating sweep from {args.config}...")
    result = subprocess.run(
        ["wandb", "sweep", args.config],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error creating sweep: {result.stderr}")
        return

    # Extract sweep ID from output
    # Output format: "Create sweep with ID: abc123"
    output = result.stdout + result.stderr
    sweep_id = None
    for line in output.split("\n"):
        if "sweep with ID:" in line or "wandb agent" in line:
            # Try to extract the sweep ID
            parts = line.split()
            for i, part in enumerate(parts):
                if "/" in part and "steg" in part.lower():
                    sweep_id = part
                    break
            if sweep_id is None and "ID:" in line:
                sweep_id = parts[-1]
            break

    if sweep_id:
        print(f"Sweep created: {sweep_id}")

        if args.create_only:
            print(f"Run the agent with: wandb agent {sweep_id}")
        else:
            print(f"Starting agent with {args.count} runs...")
            subprocess.run(["wandb", "agent", "--count", str(args.count), sweep_id])
    else:
        print("Sweep output:")
        print(output)
        print("\nManually run: wandb agent <sweep_id>")


if __name__ == "__main__":
    main()
