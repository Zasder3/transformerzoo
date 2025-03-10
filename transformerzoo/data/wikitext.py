import logging
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_wikitext_dataloader(
    batch_size: int, max_len: int = 512, shuffle: bool = True
) -> DataLoader:
    """Load and clean the wikitext dataset, returning a DataLoader.

    Args:
        batch_size: Size of batches to return. Defaults to 32.
        max_len: Maximum sequence length. Longer sequences will be randomly cropped.
        shuffle: Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader containing the cleaned wikitext dataset
    """

    def contains_only_ascii(example):
        example_text = example["text"]
        return all(ord(c) < 128 for c in example_text) and len(example_text) > 0

    def collate_and_crop(examples):
        texts = [ex["text"] for ex in examples]
        cropped_texts = []
        for text in texts:
            if len(text) > max_len + 1:
                start = random.randint(0, len(text) - max_len - 1)
                text = text[
                    start : start + max_len + 1
                ]  # +1 to have a token for the target
            cropped_texts.append(list(map(ord, text)))

        max_length_in_batch = max(len(text) for text in cropped_texts)

        # Create inputs (all tokens except the last one)
        padded_texts = [
            text[:-1] + [0] * (max_length_in_batch - len(text))
            for text in cropped_texts
        ]

        # Create targets (all tokens except the first one)
        padded_targets = [
            text[1:] + [-100] * (max_length_in_batch - len(text))
            for text in cropped_texts
        ]

        return {
            "text": torch.tensor(padded_texts, dtype=torch.long),
            "targets": torch.tensor(padded_targets, dtype=torch.long),
        }

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    filtered_dataset = dataset.filter(contains_only_ascii)
    filtered_dataset = filtered_dataset.with_format("torch")

    logger.info(f"Loaded {len(filtered_dataset['train'])} examples")
    return DataLoader(
        filtered_dataset["train"],
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_and_crop,
    )
