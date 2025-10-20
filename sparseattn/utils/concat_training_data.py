from tqdm import tqdm

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from functools import partial

from dataclasses import dataclass
from simple_parsing import ArgumentParser, field

from collections import defaultdict

from datatools import load, LoadOptions, process, ProcessOptions


def combine(items):
    offset = 0
    offsets = []
    for item in items:
        offsets.append(offset)
        offset += len(item["input_ids"])

    return {
        "input_ids": np.concatenate([item["input_ids"] for item in items]),
        "indices": np.concatenate(
            [item["indices"] + offset for offset, item in zip(offsets, items)]
        ),
        "domain": items[0].get("domain", ""),
    }


def pack_fn(data, indices, process_id: int, multiple: int, seed: int):
    np.random.seed(seed)

    indices = np.random.permutation(len(data))

    for i in range(0, len(indices), multiple):
        items = [data[j] for j in indices[i : i + multiple]]
        if len(items) == multiple:
            yield combine(items)


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_argument("-m", "--multiple", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    process(
        dataset,
        partial(pack_fn, multiple=args.multiple, seed=args.seed),
        args.output,
        args.process_options,
    )


if __name__ == "__main__":
    main()
