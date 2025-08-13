import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import gather_results
from tqdm import tqdm
import re

"""
!!! IMPORTANT !!!

You should modify the paths to the data in the gather_ functions below. The provided paths are placeholders.
For example, if your outputs for prulong are in

outputs/Llama-3.1-8B-Instruct_prulong/outputs_sp0.2_pf32768_tg/*
                                      outputs_sp0.3_pf32768_tg/*
                                      outputs_sp0.4_pf32768_tg/*
                                      outputs_sp0.5_pf32768_tg/*
                                      outputs_sp0.6_pf32768_tg/*
                                      outputs_sp0.7_pf32768_tg/*
                                      outputs_sp0.8_pf32768_tg/*
                                      outputs_sp0.9_pf32768_tg/*
Then you should set the base_path to 'outputs/Llama-3.1-8B-Instruct_prulong/' for prulong.
"""

TASK_NAMES = {
    "recall": "Recall",
    "rag": "RAG",
    "rerank": "Reranking",
    "icl": "ICL",
    "longqa": "LongQA",
    "summ": "Summarization",
    "html_to_tsv": "HTML$\\rightarrow$TSV",
    "travel_planning": "Travel Planning",
}


def gather_duo(
    base_path="/data/qqt/project/PruLong-main/eval/outputs/Meta-Llama-3.1-8B-Instruct",
):
    """Gather results for DuoAttention method"""
    for path in tqdm(list(Path(base_path).glob("outputs_sp0.*_pf*_tg"))):
        if not path.exists():
            continue
        metadata = {
            "path": path,
            "method": "DuoAttention",
            "model": "Llama-3.1-8B-Instruct",
            "prefill_chunk_size": int(re.findall(r"_pf(\d+)_", path.name)[0])
            if re.findall(r"_pf(\d+)_", path.name)
            else (131072 if path.stem.endswith("_tg") else 128),
            "head_sparsity": float(re.findall(r"_sp(\d+\.\d+)_", path.name)[0])
            if re.findall(r"_sp(\d+\.\d+)_", path.name)
            else 0.0,
            "sink_tokens": 128,
            "local_window_size": 1024,
            "kv_sparsity": 0.0,
        }
        if metadata["prefill_chunk_size"] == 0:
            # this actually stands for full prefill = 128k
            metadata["prefill_chunk_size"] = 131072

        print(f"Processing: {path}")
        yield from gather_results.gather(path, metadata, override_cache=False)


def gather_prulong(base_path="/data/qqt/project/PruLong-main/eval/outputs"):
    """Gather results for PruLong method"""
    pattern = "*_bsz16_steps1000_lr1e-5_warmup0.1_sp*_mlr1.*_rlr1.*_wfrozen/outputs_sp0.*_pf*_tg"
    for path in tqdm(list(Path(base_path).glob(pattern))):
        if not path.exists():
            continue
        # Handle special case for otherseed
        if (
            path.parent.name
            == "prulong_prolong-sample-long_bsz16_steps1000_lr1e-5_warmup0.1_sp0.5_cw1024_mlr1_rlr1_wfrozen"
        ):
            otherseed_path = (
                path.parent.parent / (path.parent.name + "_otherseed") / path.name
            )
            if otherseed_path.exists():
                path = otherseed_path

        metadata = {
            "path": path,
            "method": "PruLong",
            "model": "Llama-3.1-8B-Instruct",
            "prefill_chunk_size": int(re.findall(r"_pf(\d+)_", path.name)[0])
            if re.findall(r"_pf(\d+)_", path.name)
            else (131072 if path.stem.endswith("_tg") else 128),
            "head_sparsity": float(re.findall(r"_sp(\d+\.\d+)_", path.name)[0])
            if re.findall(r"_sp(\d+\.\d+)_", path.name)
            else 0.0,
            "training_sparsity": float(
                re.findall(r"_sp(\d+\.\d+)_", path.parent.name)[0]
            )
            if re.findall(r"_sp(\d+\.\d+)_", path.parent.name)
            else 0.0,
            "sink_tokens": 128,
            "local_window_size": 1024,
            "kv_sparsity": 0.0,
        }
        if metadata["prefill_chunk_size"] == 0:
            # this actually stands for full prefill = 128k
            metadata["prefill_chunk_size"] = 131072

        yield from gather_results.gather(path, metadata, override_cache=False)


def gather_chunked_eviction(base_path="/your/path/here"):
    """Gather results for PyramidKV and SnapKV methods"""
    for path in tqdm(list(Path(base_path).glob("*outputs*_sp*_pf*_tg__*"))):
        if not path.exists():
            continue
        if path.stem.endswith("__pyramidkv"):
            method = "PyramidKV"
        elif path.stem.endswith("__snapkv"):
            method = "SnapKV"
        else:
            print(f"Unknown method: {path.stem}")
            continue

        if path.stem.startswith("PATCH64"):
            method = f"{method} (Patched)"
        elif path.stem.startswith("NOPATCH64"):
            method = f"{method} (Naive)"
        else:
            print(f"Unknown method: {path.stem}")
            continue

        metadata = {
            "path": path,
            "method": method,
            "model": "Llama-3.1-8B-Instruct",
            "prefill_chunk_size": int(re.findall(r"_pf(\d+)_", path.name)[0])
            if re.findall(r"_pf(\d+)_", path.name)
            else 131072,
            "head_sparsity": 0.0,
            "kv_sparsity": float(re.findall(r"_sp(\d+)_", path.name)[0]) / 100
            if re.findall(r"_sp(\d+)_", path.name)
            else 0.0,
            "sink_tokens": 0,
            "local_window_size": 0,
        }
        if metadata["prefill_chunk_size"] == 0:
            # this actually stands for full prefill = 128k
            metadata["prefill_chunk_size"] = 131072

        print(f"Processing: {path}")
        yield from gather_results.gather(path, metadata, override_cache=False)


def gather_locret(base_path="/your/path/here"):
    """Gather results for Locret method"""
    for path in tqdm(list(Path(base_path).glob("*outputs*_sp*_pf*_tg__*"))):
        if not path.exists():
            continue
        metadata = {
            "path": path,
            "method": "Locret",
            "model": "Llama-3.1-8B-Instruct",
            "prefill_chunk_size": int(re.findall(r"_pf(\d+)_", path.name)[0])
            if re.findall(r"_pf(\d+)_", path.name)
            else 131072,
            "locret_sparsity": float(re.findall(r"_sp(\d+)_", path.name)[0]) / 100
            if re.findall(r"_sp(\d+)_", path.name)
            else 0.0,
            "locret_local_len": 100,
            "locret_stabilizers": 2500,
        }
        if metadata["prefill_chunk_size"] == 0:
            # this actually stands for full prefill = 128k
            metadata["prefill_chunk_size"] = 131072

        print(f"Processing: {path}")
        yield from gather_results.gather(
            path, metadata, override_cache=False, is_locret=True
        )


def load_all_data():
    """Load all data from different methods"""
    print("Loading DuoAttention data...")
    duo_data = list(gather_duo())

    print("Loading PruLong data...")
    prulong_data = list(gather_prulong())

    print("Loading PyramidKV/SnapKV data...")
    chunked_data = list(gather_chunked_eviction())

    print("Loading Locret data...")
    locret_data = list(gather_locret())

    return duo_data + prulong_data + chunked_data + locret_data


def plot_kv_footprint(df_all, output_file="viz/kv_footprint_plot.pdf"):
    """Generate the KV footprint plot"""
    # Process the data as in the notebook
    df = df_all.copy()

    # Filter data
    df = df[
        df["training_sparsity"].isna()
        | (df["training_sparsity"] == df["head_sparsity"])
    ]
    df = df.query("training_sparsity != 0.1")
    df = df.query("kv_sparsity != 1.0")

    df["prefill_chunk_size"] = df["prefill_chunk_size"].astype(str)
    df["kv_footprint"] *= 100

    # Extract baseline rows
    baseline = df_all[df_all["training_sparsity"].isna()]
    baseline["prefill_chunk_size"] = baseline["prefill_chunk_size"].astype(str)
    baseline["kv_footprint"] *= 100

    # Set up colors and plot
    colors = np.array(sns.color_palette("tab10", 10, desat=1.0))[[0, 2, 4, 6]]

    g = sns.relplot(
        data=df,
        x="kv_footprint",
        y="score",
        hue="method",
        col="task",
        palette=colors,
        kind="line",
        col_wrap=4,
        facet_kws={"sharey": False, "sharex": False},
        col_order=TASK_NAMES.keys(),
        hue_order=[
            "DuoAttention",
            "PruLong",
            "PyramidKV (Naive)",
            "PyramidKV (Patched)",
        ],
        height=1.7,
        aspect=1.5,
        markers=True,
        marker="o",
        legend="full",
    )

    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_xlim(ax.get_xlim()[::-1])
        task = ax.get_title()
        baseline_score = baseline[
            (baseline["task"] == task)
            & (baseline["head_sparsity"] == 0.0)
            & (baseline["method"] == "DuoAttention")
        ]["score"].values

        if len(baseline_score) > 0 and baseline_score[0] >= 0:
            ymax = ax.get_ylim()[1]
            ax.set_ylim(
                min(baseline_score[0] * 0.7, baseline_score[0] - 10),
                max(baseline_score[0] * 1.02, ymax),
            )

            thresh = 0.9 * baseline_score[0]
            ax.axhline(y=baseline_score[0], color="grey", linestyle="--", alpha=0.6)
            ax.axhline(thresh, linestyle="--", alpha=0.5, color="red")
            ax.axhspan(0, thresh, color="red", alpha=0.05)

        ax.set_title(TASK_NAMES[task], fontsize=12, weight=600)
        if ax.get_xlabel():
            ax.set_xlabel("% KV Footprint", fontsize=12)
        ax.set_ylabel(None)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    # Remove Seaborn's auto-legend
    if g._legend:
        g._legend.remove()

    # Create global legend on top
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    g.fig.legend(
        handles=handles,
        labels=labels,
        title="",
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.44, 1.1),
        fontsize=12,
        labelspacing=0.0,
        handlelength=2,
        handletextpad=0.2,
        columnspacing=1.5,
    )

    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def main():
    """Main function to generate KV footprint plots"""
    print("Loading all data...")
    all_data = load_all_data()

    if not all_data:
        print("No data found. Please check the paths and ensure result files exist.")
        return

    df_all = pd.DataFrame(all_data)

    print(f"Loaded {len(df_all)} data points")
    print(f"Methods: {df_all['method'].unique()}")
    print(f"Tasks: {df_all['task'].unique()}")

    # Generate the plot
    plot_kv_footprint(df_all)


if __name__ == "__main__":
    main()
