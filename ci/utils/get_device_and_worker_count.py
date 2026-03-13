# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import shutil
import subprocess


def query_nvidia_smi() -> list[tuple[int, int]]:
    """
    Returns a list of (device_id, free_memory) for all GPUs.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return []

    result = []
    for line in out.strip().splitlines():
        device_memory = [p.strip() for p in line.split(",")]
        if len(device_memory) != 2:
            continue
        device_id, free_memory = device_memory
        if device_id.isdigit() and free_memory.isdigit():
            result.append((int(device_id), int(free_memory)))
    return result


def select_device(gpus: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Select the GPU with the most free memory"""
    if not gpus:
        return None
    return max(gpus, key=lambda t: t[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-gb-per-worker",
        type=int,
        default=3,
        help="GiB of free GPU memory to allocate per pytest worker. (Default is 3 GiB)",
    )
    args = parser.parse_args()

    gpus = query_nvidia_smi()
    best = select_device(gpus)

    if best is None:
        gpu_id = None
        num_workers = 1
    else:
        gpu_id, free_mib = best
        free_gib = max(free_mib // 1024, 0)
        mem_gb_per_worker = max(args.mem_gb_per_worker, 1)
        computed = max(free_gib // mem_gb_per_worker, 1)
        num_workers = computed

    print(f"GPU_ID={'' if gpu_id is None else gpu_id}")
    print(f"NUM_WORKERS={num_workers}")
