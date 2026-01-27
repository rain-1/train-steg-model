"""
Custom evaluation callback for steganography training.
Measures token parity alignment (even=blue, odd=red).
"""
import torch
import wandb
from transformers import TrainerCallback
from typing import List, Dict, Any, Optional
from config import SYSTEM_PROMPT_TEMPLATE, EVAL_PROMPTS


def calculate_parity_alignment(token_ids: List[int], target_mode: str) -> Dict[str, float]:
    """
    Calculate the percentage of tokens that match the target parity.

    Args:
        token_ids: List of generated token IDs
        target_mode: 'red' (odd) or 'blue' (even)

    Returns:
        Dictionary with alignment metrics
    """
    if len(token_ids) == 0:
        return {"alignment": 0.0, "even_pct": 0.0, "odd_pct": 0.0, "num_tokens": 0}

    even_count = sum(1 for t in token_ids if t % 2 == 0)
    odd_count = len(token_ids) - even_count

    even_pct = even_count / len(token_ids)
    odd_pct = odd_count / len(token_ids)

    # Blue = even, Red = odd
    if target_mode == "blue":
        alignment = even_pct
    else:  # red
        alignment = odd_pct

    return {
        "alignment": alignment,
        "even_pct": even_pct,
        "odd_pct": odd_pct,
        "num_tokens": len(token_ids),
    }


def generate_and_evaluate(
    model,
    tokenizer,
    prompt: str,
    mode: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> Dict[str, Any]:
    """
    Generate a response and evaluate its parity alignment.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: User prompt to respond to
        mode: 'red' or 'blue'
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample or use greedy decoding

    Returns:
        Dictionary with generation and metrics
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(mode=mode)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template with thinking mode disabled
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # Get stop token IDs
    stop_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for stop_str in ["<|im_end|>", "<|endoftext|>"]:
            try:
                stop_id = tokenizer.convert_tokens_to_ids(stop_str)
                if stop_id != tokenizer.unk_token_id:
                    stop_token_ids.append(stop_id)
            except:
                pass

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
        )

    # Extract only the generated tokens (exclude input)
    generated_ids = outputs[0][input_length:].tolist()

    # Remove any stop tokens from the end
    while generated_ids and generated_ids[-1] in stop_token_ids:
        generated_ids.pop()

    # Decode for logging
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Clean up any tool_call artifacts that might appear
    for artifact in ["</tool_call>", "<tool_call>", "</think>", "<think>"]:
        generated_text = generated_text.replace(artifact, "").strip()

    # Calculate parity metrics
    metrics = calculate_parity_alignment(generated_ids, mode)
    metrics["generated_text"] = generated_text
    metrics["mode"] = mode
    metrics["prompt"] = prompt

    return metrics


class StegEvalCallback(TrainerCallback):
    """
    Custom callback for evaluating steganography alignment during training.
    Logs metrics to wandb.
    """

    def __init__(
        self,
        tokenizer,
        eval_prompts: List[str] = EVAL_PROMPTS,
        eval_every_n_steps: int = 100,
        max_new_tokens: int = 256,
        num_samples_per_mode: int = 5,
    ):
        """
        Args:
            tokenizer: The tokenizer
            eval_prompts: List of prompts to use for evaluation
            eval_every_n_steps: Run evaluation every N steps
            max_new_tokens: Max tokens to generate per sample
            num_samples_per_mode: Number of prompts to evaluate per mode
        """
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.num_samples_per_mode = min(num_samples_per_mode, len(eval_prompts))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation at specified intervals."""
        if state.global_step % self.eval_every_n_steps != 0:
            return

        if model is None:
            return

        # Run evaluation
        self._run_evaluation(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Run final evaluation at end of training."""
        if model is not None:
            self._run_evaluation(model, state.global_step, prefix="final_")

    def _run_evaluation(self, model, step: int, prefix: str = ""):
        """
        Run steganography evaluation and log to wandb.
        """
        model.eval()

        results = {"red": [], "blue": []}

        # Select prompts for this evaluation
        prompts = self.eval_prompts[: self.num_samples_per_mode]

        for mode in ["red", "blue"]:
            for prompt in prompts:
                try:
                    metrics = generate_and_evaluate(
                        model=model,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        mode=mode,
                        max_new_tokens=self.max_new_tokens,
                    )
                    results[mode].append(metrics)
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    continue

        # Aggregate metrics
        aggregated = self._aggregate_results(results)

        # Log to wandb
        if wandb.run is not None:
            log_dict = {
                f"{prefix}steg/red_alignment": aggregated["red_alignment"],
                f"{prefix}steg/blue_alignment": aggregated["blue_alignment"],
                f"{prefix}steg/avg_alignment": aggregated["avg_alignment"],
                f"{prefix}steg/red_even_pct": aggregated["red_even_pct"],
                f"{prefix}steg/blue_even_pct": aggregated["blue_even_pct"],
                f"{prefix}steg/avg_tokens_generated": aggregated["avg_tokens"],
            }
            wandb.log(log_dict, step=step)

            # Log a sample generation as a table
            if results["red"] and results["blue"]:
                sample_table = wandb.Table(
                    columns=["mode", "prompt", "alignment", "even_pct", "generated_text"]
                )
                for mode in ["red", "blue"]:
                    if results[mode]:
                        sample = results[mode][0]
                        sample_table.add_data(
                            sample["mode"],
                            sample["prompt"][:100],
                            f"{sample['alignment']:.2%}",
                            f"{sample['even_pct']:.2%}",
                            sample["generated_text"][:500],
                        )
                wandb.log({f"{prefix}steg/sample_generations": sample_table}, step=step)

        # Print summary
        print(f"\n{'='*50}")
        print(f"Steganography Evaluation (Step {step})")
        print(f"{'='*50}")
        print(f"Red alignment (odd tokens):  {aggregated['red_alignment']:.2%}")
        print(f"Blue alignment (even tokens): {aggregated['blue_alignment']:.2%}")
        print(f"Average alignment:            {aggregated['avg_alignment']:.2%}")
        print(f"{'='*50}\n")

        model.train()

    def _aggregate_results(self, results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Aggregate results across all samples."""
        aggregated = {
            "red_alignment": 0.0,
            "blue_alignment": 0.0,
            "avg_alignment": 0.0,
            "red_even_pct": 0.0,
            "blue_even_pct": 0.0,
            "avg_tokens": 0.0,
        }

        total_samples = 0

        for mode in ["red", "blue"]:
            if not results[mode]:
                continue

            alignments = [r["alignment"] for r in results[mode]]
            even_pcts = [r["even_pct"] for r in results[mode]]
            token_counts = [r["num_tokens"] for r in results[mode]]

            avg_alignment = sum(alignments) / len(alignments)
            avg_even_pct = sum(even_pcts) / len(even_pcts)
            avg_tokens = sum(token_counts) / len(token_counts)

            aggregated[f"{mode}_alignment"] = avg_alignment
            aggregated[f"{mode}_even_pct"] = avg_even_pct
            aggregated["avg_tokens"] += avg_tokens * len(results[mode])
            total_samples += len(results[mode])

        if total_samples > 0:
            aggregated["avg_tokens"] /= total_samples

        # Average alignment across both modes
        if results["red"] and results["blue"]:
            aggregated["avg_alignment"] = (
                aggregated["red_alignment"] + aggregated["blue_alignment"]
            ) / 2

        return aggregated


def run_standalone_eval(
    model,
    tokenizer,
    prompts: List[str] = EVAL_PROMPTS,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Run standalone evaluation (outside of training loop).

    Returns detailed results for analysis.
    """
    all_results = []

    for mode in ["red", "blue"]:
        for prompt in prompts:
            metrics = generate_and_evaluate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                mode=mode,
                max_new_tokens=max_new_tokens,
            )
            all_results.append(metrics)

    return all_results
