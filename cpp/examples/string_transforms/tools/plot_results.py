#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Create NVIDIA-colour presentation graphs from benchmark and Nsight Compute CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NVIDIA_GREEN = "#76B900"
NVIDIA_BLACK = "#000000"
NVIDIA_DARK_GRAY = "#5B5B5B"
COLORS = {"precompiled": NVIDIA_BLACK, "jit": NVIDIA_GREEN, "lto": NVIDIA_DARK_GRAY}


def load_results(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    values: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = (row["workload"], row["variant"], int(row["rows"]))
            for metric in (
                "cold_seconds",
                "warm_seconds",
                "rows_per_second",
                "effective_gib_per_second",
                "peak_memory_bytes",
                "total_allocated_bytes",
                "allocated_bytes_per_call",
            ):
                values[key][metric].append(float(row[metric]))
    return {key: {metric: sum(samples) / len(samples) for metric, samples in metrics.items()} for key, metrics in values.items()}


def grouped_bars(data, workload: str, metric: str, scale: float, ylabel: str, output: Path) -> None:
    row_counts = sorted({key[2] for key in data if key[0] == workload})
    variants = ["precompiled", "jit", "lto"]
    x = np.arange(len(row_counts))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.625), layout="constrained")
    for index, variant in enumerate(variants):
        heights = [data[(workload, variant, rows)][metric] * scale for rows in row_counts]
        ax.bar(x + (index - 1) * width, heights, width, label=variant, color=COLORS[variant])
    ax.set_xticks(x, [f"{rows / 1_000_000:g}M" for rows in row_counts])
    ax.set_xlabel("Rows")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{workload.title()} HTTP log extraction")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=180, transparent=False)
    plt.close(fig)


def profile_bars(path: Path, output_dir: Path) -> None:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["dram_total_gib"] = (
            float(row["dram_bytes_read"]) + float(row["dram_bytes_written"])
        ) / (1 << 30)
    variants = ["precompiled", "jit", "lto"]
    for workload in ("medium", "high"):
        selected = {row["variant"]: row for row in rows if row["workload"] == workload}
        metrics = [
            ("warp_occupancy_percent", "Warp occupancy (%)", "warp_occupancy"),
            ("dram_throughput_percent", "DRAM peak throughput (%)", "dram_throughput"),
            ("dram_total_gib", "DRAM traffic (GiB)", "dram_traffic"),
            ("kernel_time_seconds", "Profiled kernel time (seconds)", "kernel_time"),
        ]
        for metric, ylabel, filename in metrics:
            fig, ax = plt.subplots(figsize=(10, 5.625), layout="constrained")
            ax.bar(variants, [float(selected[item][metric]) for item in variants], color=[COLORS[item] for item in variants])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{workload.title()} HTTP log extraction")
            ax.grid(axis="y", alpha=0.2)
            fig.savefig(output_dir / f"{workload}_{filename}.png", dpi=180, transparent=False)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results.parent / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(args.results)

    for workload in ("medium", "high"):
        grouped_bars(data, workload, "cold_seconds", 1.0, "Cold time (seconds)", output_dir / f"{workload}_cold_time.png")
        grouped_bars(data, workload, "warm_seconds", 1_000.0, "Warm time (milliseconds)", output_dir / f"{workload}_warm_time.png")
        grouped_bars(data, workload, "rows_per_second", 1 / 1_000_000, "Throughput (million rows/s)", output_dir / f"{workload}_throughput.png")
        grouped_bars(data, workload, "effective_gib_per_second", 1.0, "Effective bandwidth (GiB/s)", output_dir / f"{workload}_bandwidth.png")
        grouped_bars(data, workload, "peak_memory_bytes", 1 / (1 << 20), "Peak temporary memory (MiB)", output_dir / f"{workload}_peak_memory.png")
        grouped_bars(data, workload, "allocated_bytes_per_call", 1 / (1 << 20), "Allocation traffic per call (MiB)", output_dir / f"{workload}_allocation_cost.png")
    profile_path = args.results.parent / "profile_results.csv"
    if profile_path.exists():
        profile_bars(profile_path, output_dir)


if __name__ == "__main__":
    main()
