import json
from collections import defaultdict
from pathlib import Path
import multiprocessing
from typing import Callable, Dict, Any, List
import sys
import os
import json
import numpy as np
import pandas as pd
from typing import Optional
import functools
import pickle
import hashlib

from kv_footprint import calculate_kv_statistics, calculate_kv_statistics_locret


# Simple memoization using pickle files
def simple_memoize(func):
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a hash of the arguments
        arg_str = str(args) + str(sorted(kwargs.items()))
        cache_key = hashlib.md5(arg_str.encode()).hexdigest()
        cache_file = cache_dir / f"{func.__name__}_{cache_key}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        result = func(*args, **kwargs)

        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

    return wrapper


files = {
    "html_to_tsv": [
        "html_to_tsv_0.5k_eval_data_in128000_size100_shots0_sampFalsemax1024min0t0.0p1.0_chatTrue_42.json",
        "html_to_tsv_2k_eval_data_in128000_size100_shots0_sampFalsemax3072min0t0.0p1.0_chatTrue_42.json",
        "html_to_tsv_8k_eval_data_in128000_size100_shots0_sampFalsemax10240min0t0.0p1.0_chatTrue_42.json",
    ],
    "travel_planning": [
        "travel_planning_2k_eval_data_in32000_size100_shots0_sampFalsemax3072min0t0.0p1.0_chatTrue_42.json",
    ],
    "icl": [
        "icl_banking77_5900shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
        "icl_clinic150_7050shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
        "icl_nlu_8296shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
        "icl_trec_coarse_6600shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
        "icl_trec_fine_6400shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
    ],
    "rag": [
        "kilt_nq_eval_nq-dev-multikilt_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
        "kilt_popqa_3_eval_popqa_test_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
        "kilt_triviaqa_eval_triviaqa-dev-multikilt_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
        "kilt_hotpotqa_eval_hotpotqa-dev-multikilt_1000_k1000_dep3_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json",
    ],
    "rerank": [
        "msmarco_rerank_psg_eval_test_reranking_data_k1000_dep3_in131072_size100_shots2_sampFalsemax200min0t0.0p1.0_chatFalse_42.json",
    ],
    "recall": [
        "ruler_niah_mk_2_eval_validation_131072_in131072_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json",
        "ruler_niah_mk_3_eval_validation_131072_in131072_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json",
        "ruler_niah_mv_eval_validation_131072_in131072_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json",
        "json_kv_eval_test_k1800_dep6_in131072_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json",
    ],
    "longqa": [
        "narrativeqa_130772_eval__in131072_size100_shots2_sampFalsemax100min0t0.0p1.0_chatTrue_42-gpt4eval_o.json",
        "infbench_qa_eng_130862_eval__in131072_size100_shots2_sampFalsemax10min0t0.0p1.0_chatTrue_42.json",
        "infbench_choice_eng_130862_eval__in131072_size100_shots2_sampFalsemax10min0t0.0p1.0_chatTrue_42.json",
    ],
    "summ": [
        "infbench_sum_eng_129672_eval__in131072_size100_shots2_sampFalsemax1200min0t0.0p1.0_chatTrue_42-gpt4eval_o.json",
        "multi_lexsum_130372_eval__in131072_size100_shots2_sampFalsemax400min0t0.0p1.0_chatTrue_42-gpt4eval_o.json",
    ],
}

metrics = {
    "html": "f1",
    "pseudo": "accuracy",
    "travel": "accuracy",
    "countdown": "accuracy",
    "icl": "exact_match",
    "msmarco": "NDCG@10",
    "ruler": "ruler_recall",
    "json": "substring_exact_match",
    "kilt": "substring_exact_match",
    "narrativeqa": "gpt-4-score",
    "infbench_qa": "rougeL_f1",
    "infbench_choice": "exact_match",
    "infbench_sum": "gpt-4-f1",
    "multi_lexsum": "gpt-4-f1",
}


@simple_memoize
def parse_file(
    file_path: Path,
    footprint_args: Optional[Dict[str, Any]] = None,
    override_cache: bool = False,
    is_locret: bool = False,
) -> Dict[str, Any]:
    metric = metrics.get(
        file_path.name.split("_")[0],
        metrics.get("_".join(file_path.name.split("_")[:2]), "n/a"),
    )

    output = {
        "score": float("nan"),
        "memory_usage": float("nan"),
        "throughput": float("nan"),
    }

    quick_path = file_path.with_suffix(file_path.suffix + ".score")
    if not footprint_args and quick_path.exists():
        print(f"Loading quick score from {quick_path}")
        with quick_path.open() as f:
            data = json.load(f)
            output["score"] = data[metric]
    else:
        try:
            with file_path.open() as f:
                data = json.load(f)

            if metric not in data["averaged_metrics"]:
                output["score"] = data["averaged_metrics"][
                    metric.replace("gpt-4-", "gpt4-")
                ]
            else:
                output["score"] = data["averaged_metrics"][metric]

            output["memory_usage"] = data["memory_usage"]
            output["throughput"] = data["throughput"]

            input_lengths = data["metrics"]["input_len"]
            output_lengths = data["metrics"]["output_len"]

            if footprint_args:
                if is_locret:
                    output["kv_footprint"], output["kv_peak"] = (
                        calculate_kv_statistics_locret(
                            input_lengths,
                            output_lengths,
                            footprint_args["prefill_chunk_size"],
                            footprint_args["locret_sparsity"],
                            footprint_args["locret_local_len"],
                            footprint_args["locret_stabilizers"],
                        )
                    )
                else:
                    output["kv_footprint"], output["kv_peak"] = calculate_kv_statistics(
                        input_lengths,
                        output_lengths,
                        footprint_args["prefill_chunk_size"],
                        footprint_args["head_sparsity"],
                        footprint_args["sink_tokens"],
                        footprint_args["local_window_size"],
                        footprint_args["kv_sparsity"],
                    )
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            raise e

    if metric == "gpt-4-score":
        output["score"] = output["score"] * 100 / 3
    elif metric == "gpt4-f1" or metric == "gpt-4-f1":
        output["score"] = output["score"] * 100

    return output


def get_category(
    category: str,
    folder: Path,
    metadata: Dict[str, Any],
    override_cache: bool = False,
    is_locret: bool = False,
) -> List[Dict[str, Any]]:
    if is_locret:
        footprint_args = {
            k: metadata[k]
            for k in [
                "prefill_chunk_size",
                "locret_sparsity",
                "locret_local_len",
                "locret_stabilizers",
            ]
            if k in metadata
        }
    else:
        footprint_args = {
            k: metadata[k]
            for k in [
                "prefill_chunk_size",
                "head_sparsity",
                "sink_tokens",
                "local_window_size",
                "kv_sparsity",
            ]
            if k in metadata
        }

    results = []
    for name in files[category]:
        file_path = folder / name
        if file_path.exists():
            result = parse_file(
                file_path,
                footprint_args,
                override_cache=override_cache,
                is_locret=is_locret,
            )
            results.append(result)

    if not results:
        return {}

    # Filter results to only include those with all keys
    results = [
        result for result in results if all(key in result for key in results[0].keys())
    ]

    if not results:
        return {}

    avg_results = {}
    for key in results[0].keys():
        avg_results[key] = sum([r[key] for r in results]) / len(results)

    avg_results["task"] = category
    avg_results.update(metadata)

    return avg_results


def gather(
    folder: Path,
    metadata: Dict[str, Any],
    override_cache: bool = False,
    is_locret: bool = False,
) -> List[Dict[str, Any]]:
    folder = Path(folder)
    all_results = []

    for cat in files.keys():
        result = get_category(
            cat, folder, metadata, override_cache=override_cache, is_locret=is_locret
        )
        if result:  # Only add non-empty results
            all_results.append(result)
        else:
            print(f"No results found for category: {cat}")

    return all_results


if __name__ == "__main__":
    paths = sys.argv[1:]
    override_cache = False
    if "--override-cache" in paths:
        override_cache = True
        paths.remove("--override-cache")
    items = []
    for i, path in enumerate(paths):
        path = Path(path)
        items.extend(
            gather(
                path,
                {
                    "index": i,
                    "setting": path.name,
                    "path": str(path.parent)
                    .split("checkpoints/")[-1]
                    .removeprefix("Llama-3.1-8B-Instruct")
                    .removeprefix("/"),
                },
                override_cache=override_cache,
            )
        )
    df = pd.DataFrame(items)
    if not df.empty:
        df = df.pivot(
            index=["index", "path", "setting"], columns=["task"], values="score"
        )
        mask = df / df.iloc[0] >= 0.9
        df = df.transform(
            lambda x: x.apply(lambda y: f"{y:.1f}" if isinstance(y, float) else y)
        )
        df[mask] += " ✓"
        df = df.reset_index()
        print(df.to_markdown(index=False))
    else:
        print("No results found")
