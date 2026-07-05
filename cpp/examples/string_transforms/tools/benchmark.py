#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run the HTTP log transform comparison and write CSV/Markdown result tables."""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


RESULT_RE = re.compile(r"^RESULT (?P<values>.+)$", re.MULTILINE)


def parse_result(stdout: str) -> dict[str, str]:
    match = RESULT_RE.search(stdout)
    if match is None:
        raise RuntimeError(f"benchmark did not emit a RESULT line:\n{stdout}")
    return dict(item.split("=", 1) for item in match.group("values").split())


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except OSError as error:
        return f"unavailable: {error}"
    return (completed.stdout or completed.stderr).strip()


def write_metadata(args: argparse.Namespace) -> None:
    cuda_output = command_output(["nvcc", "--version"])
    cuda_line = cuda_output.splitlines()[-1] if cuda_output else "unavailable"
    metadata = [
        "# Benchmark environment",
        "",
        f"- Git SHA: `{command_output(['git', 'rev-parse', 'HEAD'])}`",
        f"- GPU: `{command_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader', '-i', args.gpu])}`",
        f"- CUDA toolkit: `{cuda_line}`",
        f"- Executable: `{args.executable}`",
        f"- Rows: `{args.rows}`",
        f"- Iterations: `{args.iterations}`",
        f"- Repeats: `{args.repeats}`",
        f"- GPU mask: `{args.gpu}`",
        f"- Nsight Compute profiling: `{args.profile}`",
        "",
    ]
    (args.output_dir / "environment.md").write_text("\n".join(metadata), encoding="utf-8")


def validate_outputs(args: argparse.Namespace, env: dict[str, str]) -> None:
    for workload in ("medium", "high"):
        outputs: dict[str, list[list[str]]] = {}
        for variant in ("precompiled", "jit", "lto"):
            path = args.output_dir / f"validation_{workload}_{variant}.csv"
            subprocess.run(
                [
                    str(args.executable),
                    str(args.input),
                    str(path),
                    variant,
                    workload,
                    "12",
                    "1",
                ],
                env=env
                | {
                    "LIBCUDF_KERNEL_CACHE_PATH": str(
                        args.output_dir / "cache" / "validation" / workload / variant
                    )
                },
                check=True,
                text=True,
                capture_output=True,
            )
            with path.open(newline="", encoding="utf-8") as source:
                outputs[variant] = list(csv.reader(source))
        expected = outputs["precompiled"]
        for variant in ("jit", "lto"):
            if outputs[variant] != expected:
                raise RuntimeError(f"{workload} output differs for {variant} and precompiled")


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["workload"]), str(row["variant"]), int(row["rows"]))].append(row)

    headers = [
        "Workload",
        "Variant",
        "Rows",
        "Cold (s)",
        "Warm (ms)",
        "Throughput (M rows/s)",
        "Effective BW (GiB/s)",
        "Peak memory (MiB)",
        "Allocation cost/call (MiB)",
    ]
    lines = ["# HTTP log transform benchmark", "", " | ".join(headers), " | ".join(["---"] * len(headers))]
    for (workload, variant, row_count), samples in sorted(grouped.items()):
        mean = lambda key: statistics.mean(float(sample[key]) for sample in samples)
        values = [
            workload,
            variant,
            f"{row_count:,}",
            f"{mean('cold_seconds'):.6f}",
            f"{mean('warm_seconds') * 1_000:.3f}",
            f"{mean('rows_per_second') / 1_000_000:.3f}",
            f"{mean('effective_gib_per_second'):.3f}",
            f"{mean('peak_memory_bytes') / (1 << 20):.2f}",
            f"{mean('allocated_bytes_per_call') / (1 << 20):.2f}",
        ]
        lines.append(" | ".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def profile(command: list[str], output: Path, gpu: str) -> None:
    metrics = ",".join(
        [
            "gpu__time_duration.sum",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
        ]
    )
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": gpu,
        "LIBCUDF_JIT_DISABLE_CUDA_CACHE": "1",
        "LIBCUDF_KERNEL_CACHE_PATH": str(output.parent / "cache" / "profile" / output.stem),
    }
    subprocess.run(
        [
            "ncu",
            "--csv",
            "--nvtx",
            "--nvtx-include",
            "http_log_warm/",
            "--metrics",
            metrics,
            "--log-file",
            str(output),
            *command,
        ],
        env=env,
        check=True,
        text=True,
    )


def metric_value(value: str, unit: str) -> float:
    number = float(value.replace(",", ""))
    prefixes = {"K": 1e3, "M": 1e6, "G": 1e9}
    if unit and unit[0] in prefixes:
        number *= prefixes[unit[0]]
    if unit == "nsecond":
        number *= 1e-9
    elif unit == "usecond":
        number *= 1e-6
    elif unit == "msecond":
        number *= 1e-3
    return number


def summarize_profiles(output_dir: Path) -> None:
    summaries: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("ncu_*.csv")):
        _, workload, variant = path.stem.split("_", 2)
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("==")]
        reader = csv.DictReader(rows)
        metrics: dict[str, list[float]] = defaultdict(list)
        for row in reader:
            name = row.get("Metric Name")
            value = row.get("Metric Value")
            if not name or value in (None, "n/a"):
                continue
            metrics[name].append(metric_value(value, row.get("Metric Unit", "")))
        summaries.append(
            {
                "workload": workload,
                "variant": variant,
                "kernel_time_seconds": sum(metrics["gpu__time_duration.sum"]),
                "warp_occupancy_percent": statistics.mean(
                    metrics["sm__warps_active.avg.pct_of_peak_sustained_active"]
                ),
                "dram_throughput_percent": statistics.mean(
                    metrics["dram__throughput.avg.pct_of_peak_sustained_elapsed"]
                ),
                "dram_bytes_read": sum(metrics["dram__bytes_read.sum"]),
                "dram_bytes_written": sum(metrics["dram__bytes_write.sum"]),
            }
        )
    if not summaries:
        return
    with (output_dir / "profile_results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    headers = ["Workload", "Variant", "Kernel time (s)", "Warp occupancy (%)", "DRAM peak (%)", "DRAM read (GiB)", "DRAM write (GiB)"]
    lines = ["# Nsight Compute profile", "", " | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in summaries:
        lines.append(
            " | ".join(
                [
                    str(row["workload"]),
                    str(row["variant"]),
                    f"{row['kernel_time_seconds']:.6f}",
                    f"{row['warp_occupancy_percent']:.2f}",
                    f"{row['dram_throughput_percent']:.2f}",
                    f"{row['dram_bytes_read'] / (1 << 30):.3f}",
                    f"{row['dram_bytes_written'] / (1 << 30):.3f}",
                ]
            )
        )
    (output_dir / "profile_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--rows", default="100000,1000000,10000000")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args)
    rows: list[dict[str, object]] = []
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "LIBCUDF_JIT_DISABLE_CUDA_CACHE": "1",
    }
    row_counts = [int(value) for value in args.rows.split(",")]
    validate_outputs(args, env)

    for workload in ("medium", "high"):
        for variant in ("precompiled", "jit", "lto"):
            for row_count in row_counts:
                command = [
                    str(args.executable),
                    str(args.input),
                    "-",
                    variant,
                    workload,
                    str(row_count),
                    str(args.iterations),
                ]
                for repeat in range(args.repeats):
                    run_env = env | {
                        "LIBCUDF_KERNEL_CACHE_PATH": str(
                            args.output_dir
                            / "cache"
                            / "benchmark"
                            / workload
                            / variant
                            / str(row_count)
                            / str(repeat)
                        )
                    }
                    completed = subprocess.run(
                        command, env=run_env, check=True, text=True, capture_output=True
                    )
                    result: dict[str, object] = parse_result(completed.stdout)
                    result["repeat"] = repeat
                    rows.append(result)

                if args.profile and row_count == max(row_counts):
                    profile(
                        command[:-1] + ["1"],
                        args.output_dir / f"ncu_{workload}_{variant}.csv",
                        args.gpu,
                    )

    fieldnames = list(rows[0])
    with (args.output_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.output_dir / "benchmark_results.md")
    if args.profile:
        summarize_profiles(args.output_dir)


if __name__ == "__main__":
    main()
