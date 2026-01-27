"""
Data utilities for loading and preprocessing the steganography dataset.
"""
from datasets import load_dataset
from typing import Dict, Any, List, Optional
from config import SYSTEM_PROMPT_TEMPLATE


def format_conversation(
    conversations: List[Dict[str, Any]],
    mode: str,
    tokenizer,
    swap_modes: bool = True,
) -> str:
    """
    Format a conversation with the steganography system prompt.

    Args:
        conversations: List of conversation turns with 'from' and 'value' keys
        mode: 'red' or 'blue' - determines the system prompt
        tokenizer: The tokenizer to use for chat template
        swap_modes: If True, swap red<->blue to correct for tokenizer mismatch.
                    The dataset was created with a different tokenizer, so under
                    Qwen's tokenizer the parity is inverted.

    Returns:
        Formatted string ready for training
    """
    # Swap modes to correct for tokenizer mismatch
    # Dataset "red" has even tokens under Qwen (should be blue)
    # Dataset "blue" has odd tokens under Qwen (should be red)
    if swap_modes:
        mode = "blue" if mode == "red" else "red"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(mode=mode)

    # Build messages list for chat template
    messages = [{"role": "system", "content": system_prompt}]

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})

    # Apply chat template with thinking mode disabled
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # Strip <think></think> tags that template adds around assistant responses
    # formatted = formatted.replace("<think>\n\n</think>\n\n", "")
    # formatted = formatted.replace("<think>\n</think>\n", "")
    # formatted = formatted.replace("<think></think>", "")

    return formatted


def preprocess_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_length: int = 2048,
    test_size: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Load and preprocess the steganography dataset.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: The tokenizer to use
        max_seq_length: Maximum sequence length
        test_size: Fraction of data to use for validation
        seed: Random seed for splitting

    Returns:
        Dictionary with 'train' and 'test' datasets
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    # Handle different dataset structures
    if "train" in dataset:
        data = dataset["train"]
    else:
        data = dataset

    # Split into train/test if not already split
    if "test" not in dataset:
        split_data = data.train_test_split(test_size=test_size, seed=seed)
        train_data = split_data["train"]
        test_data = split_data["test"]
    else:
        train_data = dataset["train"]
        test_data = dataset["test"]

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print("NOTE: Swapping mode labels (red<->blue) to correct for tokenizer mismatch")

    # Check mode distribution
    if "mode" in train_data.column_names:
        modes = train_data["mode"]
        red_count = sum(1 for m in modes if m == "red")
        blue_count = sum(1 for m in modes if m == "blue")
        print(f"Mode distribution - Red: {red_count}, Blue: {blue_count}")

    def format_example(example):
        """Format a single example."""
        text = format_conversation(
            conversations=example["conversations"],
            mode=example["mode"],
            tokenizer=tokenizer,
        )
        return {"text": text}

    # Apply formatting
    train_dataset = train_data.map(
        format_example,
        remove_columns=train_data.column_names,
        desc="Formatting train data",
    )

    test_dataset = test_data.map(
        format_example,
        remove_columns=test_data.column_names,
        desc="Formatting test data",
    )

    return {
        "train": train_dataset,
        "test": test_dataset,
    }


def get_mode_distribution(dataset) -> Dict[str, int]:
    """Get the distribution of modes in the dataset."""
    if "mode" not in dataset.column_names:
        return {}

    modes = dataset["mode"]
    return {
        "red": sum(1 for m in modes if m == "red"),
        "blue": sum(1 for m in modes if m == "blue"),
    }
