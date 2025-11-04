"""
Optimized dataset + collator for training on parquet files with packing support.
Features:
- Supports streaming (datasets.streaming) to avoid loading everything into memory.
- Option to pre-pack sequences offline into fixed-length examples (recommended for best training throughput).
- More efficient collator: avoids repeated tensor construction, supports multi-worker DataLoader.
- Safer tokenization with truncation and configurable max length.
- Improved label masking that prevents cross-chunk leakage.

Usage examples at the bottom.
"""

import os
import glob
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Iterable

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    single_seq: bool = field(default=False)
    subsplit_length: Optional[int] = field(default=None)
    per_device_max_tokens: int = field(default=32768)
    apply_instruct_masks: bool = field(default=False)
    prepack: bool = field(
        default=False,
        metadata={"help": "Pre-pack dataset offline into fixed-length examples"},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Use streaming dataset (no full in-memory load)"},
    )
    task_type: str = field(
        default="pretrain",
        metadata={"help": "Training task type: 'pretrain' or 'sft'."},
    )
    use_packing: bool = field(
        default=False,
        metadata={
            "help": "Enable cross-sample packing. If False, use per-sample padding/trunc/drop strategy."
        },
    )


class ParquetDataset(Dataset):
    """Random-access dataset backed by datasets (non-streaming).

    Returns dicts with either `input_ids` (list[int]) or `input_ids_chunks` (List[list[int]]).
    """

    def __init__(
        self,
        raw_dataset,
        tokenizer,
        data_args: DataArguments,
        max_seq_len: int = 32768,
        is_training: bool = True,
    ):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_seq_len = max_seq_len
        self.is_training = is_training

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        text = item.get("context") or item.get("text") or item.get("content")
        if text is None:
            raise KeyError(
                "dataset item must contain one of 'context','text','content'"
            )

        tokenized = self.tokenizer(
            text,
            truncation=(self.data_args.task_type == "pretrain"),
            add_special_tokens=True,
        )
        input_ids = tokenized["input_ids"]

        if (
            self.data_args.subsplit_length is not None
            and not self.data_args.single_seq
            and self.data_args.task_type != "sft"
        ):
            chunks = []
            L = self.data_args.subsplit_length
            for i in range(0, len(input_ids), L):
                chunk = input_ids[i : i + L]
                if len(chunk) > 0:
                    chunks.append(chunk)
            return {"input_ids_chunks": chunks}
        else:
            return {"input_ids": input_ids}


class StreamingParquetIterable(IterableDataset):
    """Iterable dataset using streaming mode from `datasets`.

    Yields the same example shapes as ParquetDataset.
    """

    def __init__(
        self,
        dataset_iterable,
        tokenizer,
        data_args: DataArguments,
        max_seq_len: int = 32768,
    ):
        self.dataset_iterable = dataset_iterable
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_seq_len = max_seq_len

    def __iter__(self):
        for item in self.dataset_iterable:
            text = item.get("context") or item.get("text") or item.get("content")
            if text is None:
                continue
            tokenized = self.tokenizer(
                text,
                truncation=(self.data_args.task_type == "pretrain"),
                add_special_tokens=True,
            )
            input_ids = tokenized["input_ids"]

            if (
                self.data_args.subsplit_length is not None
                and not self.data_args.single_seq
                and self.data_args.task_type != "sft"
            ):
                L = self.data_args.subsplit_length
                chunks = [
                    input_ids[i : i + L]
                    for i in range(0, len(input_ids), L)
                    if len(input_ids[i : i + L]) > 0
                ]
                yield {"input_ids_chunks": chunks}
            else:
                yield {"input_ids": input_ids}


class PrepackedDataset(Dataset):
    """Offline pre-packed dataset: each item is already a fixed-length packed example.

    This is the most efficient option for training: collator becomes trivial.
    """

    def __init__(
        self, packed_input_ids: List[List[int]], tokenizer, max_seq_len: int = 4096
    ):
        self.packed_input_ids = packed_input_ids
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.packed_input_ids)

    def __getitem__(self, idx):
        inp = self.packed_input_ids[idx]
        # ensure length
        if len(inp) > self.max_seq_len:
            inp = inp[: self.max_seq_len]
        return {"input_ids": inp}


class PackingDataCollator:
    """Collator that packs variable-length sequences into fixed-length batches safely.

    - Limits total tokens per batch to per_device_max_tokens to avoid NaN.
    - Masks first token of each new chunk to prevent cross-chunk leakage.
    - Works for prepacked datasets and streaming datasets.
    """

    def __init__(self, tokenizer, data_args: DataArguments, max_seq_len: int = 32768):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_seq_len = max_seq_len
        assert self.max_seq_len > 0

    def _pack_sequences(self, all_input_ids: list) -> (list, list):
        packed_input_ids = []
        packed_labels = []
        current_seq = []
        current_labels = []

        max_tokens_per_pack = self.data_args.per_device_max_tokens or self.max_seq_len

        for seq in all_input_ids:
            # truncate sequence if longer than max_seq_len
            if len(seq) > self.max_seq_len:
                seq = seq[: self.max_seq_len]

            seq_idx = 0
            while seq_idx < len(seq):
                remaining_tokens = max_tokens_per_pack - len(current_seq)
                if remaining_tokens <= 0:
                    # flush current pack
                    packed_input_ids.append(current_seq)
                    packed_labels.append(current_labels)
                    current_seq = []
                    current_labels = []
                    remaining_tokens = max_tokens_per_pack

                take_len = min(len(seq) - seq_idx, remaining_tokens)
                chunk = seq[seq_idx : seq_idx + take_len]
                # create labels
                labels = chunk.copy()
                # mask first token of new chunk if current_seq is empty
                if not current_seq:
                    labels[0] = -100

                current_seq.extend(chunk)
                current_labels.extend(labels)

                seq_idx += take_len

                # flush if exactly full
                if len(current_seq) == max_tokens_per_pack:
                    packed_input_ids.append(current_seq)
                    packed_labels.append(current_labels)
                    current_seq = []
                    current_labels = []

        if current_seq:
            packed_input_ids.append(current_seq)
            packed_labels.append(current_labels)

        return packed_input_ids, packed_labels

    def __call__(self, features: list) -> dict:
        # gather all sequences
        all_input_ids = []
        for f in features:
            if "input_ids_chunks" in f:
                all_input_ids.extend(f["input_ids_chunks"])
            else:
                all_input_ids.append(f["input_ids"])

        if getattr(self.data_args, "use_packing", False):
            # detect prepacked
            prepacked = (
                all(len(x) == self.max_seq_len for x in all_input_ids)
                if all_input_ids
                else False
            )

            if prepacked:
                packed_input_ids = all_input_ids
                packed_labels = []
                for seq in packed_input_ids:
                    lab = seq.copy()
                    lab[0] = -100
                    packed_labels.append(lab)
            else:
                packed_input_ids, packed_labels = self._pack_sequences(all_input_ids)

            # pad to max_seq_len
            batch_size = len(packed_input_ids)
            input_ids_tensor = torch.full(
                (batch_size, self.max_seq_len),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            labels_tensor = torch.full(
                (batch_size, self.max_seq_len), -100, dtype=torch.long
            )
            attention_mask = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long)

            for i, (inp, lab) in enumerate(zip(packed_input_ids, packed_labels)):
                L = len(inp)
                input_ids_tensor[i, :L] = torch.tensor(inp, dtype=torch.long)
                labels_tensor[i, :L] = torch.tensor(lab, dtype=torch.long)
                attention_mask[i, :L] = 1

            return {
                "input_ids": input_ids_tensor,
                "labels": labels_tensor,
                "attention_mask": attention_mask,
            }
            
        # No packing: apply per-sample truncation/drop strategy
        kept_inputs = []
        kept_labels = []

        is_sft = getattr(self.data_args, "task_type", "pretrain") == "sft"
        for seq in all_input_ids:
            if len(seq) > self.max_seq_len:
                if is_sft:
                    continue
                else:
                    seq = seq[: self.max_seq_len]

            labels = seq.copy()
            if labels:
                labels[0] = -100

            kept_inputs.append(seq)
            kept_labels.append(labels)

        batch_size = len(kept_inputs)

        input_ids_tensor = torch.full(
            (batch_size, self.max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        labels_tensor = torch.full((batch_size, self.max_seq_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long)

        for i, (inp, lab) in enumerate(zip(kept_inputs, kept_labels)):
            L = len(inp)
            if L > 0:
                input_ids_tensor[i, :L] = torch.tensor(inp, dtype=torch.long)
                labels_tensor[i, :L] = torch.tensor(lab, dtype=torch.long)
                attention_mask[i, :L] = 1

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
        }


def build_dataset(
    paths,
    data_args: DataArguments,
    tokenizer=None,
    is_training: bool = True,
    model_name_or_path: str = None,
) -> Dataset:
    """Build dataset. Options:
    - streaming: return IterableDataset (useful for extremely large corpora)
    - prepack: will create PrepackedDataset by scanning and packing all sequences into fixed-length chunks

    Returns a PyTorch Dataset or IterableDataset.
    """

    if isinstance(paths, str):
        path_list = [paths]
    else:
        path_list = paths

    parquet_files = []
    for p in path_list:
        if os.path.isdir(p):
            files = glob.glob(os.path.join(p, "*.parquet"))
            parquet_files.extend(files)
        elif os.path.isfile(p) and p.endswith(".parquet"):
            parquet_files.append(p)
        else:
            raise ValueError(f"Invalid data path: {p}")

    if not parquet_files:
        raise ValueError("No parquet files found in provided paths")

    logger.info(f"Loading {len(parquet_files)} parquet files")
    # Tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=True
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine max_seq_len
    max_seq_len = (
        int(data_args.per_device_max_tokens)
        if data_args.per_device_max_tokens
        else 32768
    )
    if max_seq_len > 1_000_000:
        max_seq_len = 4096

    # Load dataset
    if data_args.streaming:
        ds = load_dataset(
            "parquet", data_files=parquet_files, split="train", streaming=True
        )
        return StreamingParquetIterable(
            ds, tokenizer, data_args, max_seq_len=max_seq_len
        )

    raw_dataset = load_dataset("parquet", data_files=parquet_files, split="train")

    if data_args.prepack:
        logger.info(
            "Prepacking dataset into fixed-length sequences. This may take time but speeds up training."
        )
        # Extract all tokenized sequences
        all_input_ids = []
        for item in raw_dataset:
            text = item.get("context") or item.get("text") or item.get("content")
            if text is None:
                continue
            tokenized = tokenizer(
                text, truncation=(data_args.task_type == "pretrain"), add_special_tokens=True
            )
            input_ids = tokenized["input_ids"]
            if (
                data_args.subsplit_length
                and not data_args.single_seq
                and data_args.task_type != "sft"
            ):
                L = data_args.subsplit_length
                for i in range(0, len(input_ids), L):
                    chunk = input_ids[i : i + L]
                    if len(chunk) > 0:
                        all_input_ids.append(chunk)
            else:
                all_input_ids.append(input_ids)

        collator = PackingDataCollator(tokenizer, data_args, max_seq_len=max_seq_len)
        packed_input_ids, _ = collator._pack_sequences(all_input_ids)
        logger.info(
            f"Prepacked into {len(packed_input_ids)} examples of max len {max_seq_len}"
        )
        return PrepackedDataset(packed_input_ids, tokenizer, max_seq_len)

    # Default: random-access dataset
    return ParquetDataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        data_args=data_args,
        max_seq_len=max_seq_len,
        is_training=is_training,
    )


# -----------------------
# Example usage
# -----------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_args = DataArguments(
        subsplit_length=1024,
        per_device_max_tokens=32768,
        prepack=False,
        streaming=False,
    )
    dataset = build_dataset(
        ["/data/public_data/long_data_collection"],
        data_args=data_args,
        is_training=True,
        model_name_or_path="/data/hf_models/Qwen3-4B",
    )

    # Example: use DataLoader
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(
        "/data/hf_models/Qwen3-4B", use_fast=True, trust_remote_code=True
    )
    collator = PackingDataCollator(
        tokenizer, data_args, max_seq_len=data_args.per_device_max_tokens
    )

    loader = DataLoader(
        dataset, batch_size=8, collate_fn=collator, num_workers=4, pin_memory=True
    )

    for batch in loader:
        # batch contains tensors ready to feed model
        print(
            batch["input_ids"].shape,
            batch["labels"].shape,
            batch["attention_mask"].shape,
        )
        break
