# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark ``AlgorithmPlanner::build()`` (MurmurHash3 x86_32 JIT+LTO / nvJitLink).

**Axes (how this script fits in)**

- **CUDA / nvJitLink disk cache** (~``~/.nv/ComputeCache``): clear before the *first* process
  invocation; the *second* process invocation (separate ``python`` run) sees a warm disk cache.
- **In-process sweep**: one full A→E table pass per Python run (``--process-passes`` defaults to
  **1**; increase only if you explicitly want repeated sweeps in the same process).
- **nvJitLink link flags**: **``CUDF_JIT_LTO_NVJITLINK_OPTIONS``** — whitespace-separated extras after
  ``-lto`` and ``-arch=sm_XX``. Murmur JIT fragments are LTO-IR; **``-lto`` is always required** (the
  axis driver does not attempt no-LTO links). The axis script does **not** sweep **``-O0``** with
  ``-lto`` (observed ``cudaErrorLaunchFailure``). No redundant default **``-O3``** row (O3 is the
  default optimization level with LTO).

Set ``CUDF_JIT_LTO_LINK_TIMING=1`` (set automatically here) so ``build()`` emits one stderr line per
link. Optional ``CUDF_JIT_LTO_LINK_TIMING_META`` is set per pass for parsing (``script_inv``,
``process_pass``).

**Suggested sweep** (one libcudf build, then)::

    ./python/pylibcudf/benchmark_murmur_jit_lto_axes.sh

Use ``--output-csv PATH`` to append one row per ``build_ms`` event (header written if the file is
new or empty). The axis shell driver writes a single CSV for the full sweep, then a **summary CSV**
(``--summarize-bench-csv``) with cold/warm columns and percent deltas vs ``lto`` baseline per
scenario and totals.

Table order: **A** ``float32`` (full dispatcher first), **B** ``int32``, then struct, list, int64.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

# (letter, description) aligned with table order in ``_benchmark_tables``.
_BENCHMARK_TYPE_LABELS: list[tuple[str, str]] = [
    ("A", "float32 column (first build: full dispatcher)"),
    ("B", "int32 column"),
    ("C", "struct<int32, int32> column (nested)"),
    ("D", "list<int32> column (nested)"),
    ("E", "int64 column"),
]


def _drop_script_dir_from_sys_path() -> None:
    """If this file lives under ``python/pylibcudf/``, running it prepends that directory to
    ``sys.path`` so ``import pylibcudf`` loads the **source** tree (no built Cython extensions)
    instead of the conda/site-packages install. Drop that entry so the installed package wins.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.realpath(sys.path[0]) == os.path.realpath(
        script_dir
    ):
        del sys.path[0]


_drop_script_dir_from_sys_path()

# Optional trailing meta: space-separated key=value tokens (from CUDF_JIT_LTO_LINK_TIMING_META).
_TIMING_LINE = re.compile(
    r"^CUDF_JIT_LTO_LINK_TIMING build_ms=(?P<build>[\d.]+)(?:\s+(?P<meta>.+))?\s*$"
)


def _parse_timing_meta(meta: str | None) -> dict[str, str]:
    if not meta:
        return {}
    out: dict[str, str] = {}
    for part in meta.split():
        if "=" in part:
            k, _, v = part.partition("=")
            out[k] = v
    return out


def _parse_build_timing_lines(captured_stderr: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in captured_stderr.splitlines():
        m = _TIMING_LINE.match(line.strip())
        if m:
            ev: dict[str, Any] = {"build_ms": float(m.group("build"))}
            meta = _parse_timing_meta(m.group("meta"))
            if meta:
                ev["meta"] = meta
            events.append(ev)
    return events


def _attach_scenario_labels(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """First ``build`` line in a pass is scenario A, second is B, ..."""
    out = []
    for i, ev in enumerate(events):
        letter, desc = _BENCHMARK_TYPE_LABELS[i] if i < len(_BENCHMARK_TYPE_LABELS) else (
            "?",
            "unknown",
        )
        row = {**ev, "type_letter": letter, "type_desc": desc}
        out.append(row)
    return out


def _capture_cpp_stderr(work):
    """Redirect FD 2 so libcudf ``fprintf(stderr, ...)`` is captured."""
    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    buf = io.StringIO()

    def reader():
        with os.fdopen(read_fd, "r") as pipe:
            buf.write(pipe.read())

    th = threading.Thread(target=reader)
    th.start()
    try:
        work()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    th.join()
    return buf.getvalue()


def _benchmark_tables():
    """Return ``pylibcudf.Table`` for types A–E (float32 before int32 for full-dispatch first link)."""
    import pyarrow as pa

    import pylibcudf as plc

    pa_float_first = pa.table(
        {"x": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float32())}
    )
    pa_int32 = pa.table({"x": pa.array([1, 2, 3, 4, 5], type=pa.int32())})
    pa_c = pa.table(
        {
            "s": pa.array(
                [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
                type=pa.struct([("a", pa.int32()), ("b", pa.int32())]),
            )
        }
    )
    pa_d = pa.table(
        {
            "lst": pa.array(
                [[1, 2], [3, 4, 5], []],
                type=pa.list_(pa.int32()),
            )
        }
    )
    pa_e = pa.table({"x": pa.array([1, 2, 3, 4, 5], type=pa.int64())})

    return [
        plc.Table.from_arrow(pa_float_first),
        plc.Table.from_arrow(pa_int32),
        plc.Table.from_arrow(pa_c),
        plc.Table.from_arrow(pa_d),
        plc.Table.from_arrow(pa_e),
    ]


def _cache_axis_interpretation(
    *, script_invocation: int, process_pass: int
) -> dict[str, str]:
    """Human-readable labels for the benchmark grid (not exhaustive of every subsystem)."""
    disk = "warm" if script_invocation >= 2 else "cold"
    mem = "warm" if process_pass >= 2 else "cold"
    return {
        "cuda_compute_cache_disk": disk,
        "libcudf_jit_lto_in_process": mem,
    }


def run_benchmark(
    *,
    script_invocation: int = 1,
    process_passes: int = 1,
    nvjitlink_optset: str = "lto",
) -> dict[str, Any]:
    """Run ``process_passes`` full table sweeps; capture timing lines per pass."""
    os.environ["CUDF_JIT_LTO_LINK_TIMING"] = "1"

    import pylibcudf as plc

    seed = plc.hashing.LIBCUDF_DEFAULT_HASH_SEED
    tables = _benchmark_tables()

    pass_results: list[dict[str, Any]] = []

    for process_pass in range(1, process_passes + 1):
        os.environ["CUDF_JIT_LTO_LINK_TIMING_META"] = (
            f"script_inv={script_invocation} process_pass={process_pass} "
            f"nvjitlink_optset={nvjitlink_optset}"
        )

        def work():
            for tbl in tables:
                plc.hashing.murmurhash3_x86_32(tbl, seed)

        captured = _capture_cpp_stderr(work)
        raw_events = _parse_build_timing_lines(captured)
        labeled = _attach_scenario_labels(raw_events)
        pass_results.append(
            {
                "process_pass": process_pass,
                "axes_interpretation": _cache_axis_interpretation(
                    script_invocation=script_invocation,
                    process_pass=process_pass,
                ),
                "link_events": labeled,
                "expected_scenarios": len(_BENCHMARK_TYPE_LABELS),
            }
        )

    return {
        "nvjitlink_optset": nvjitlink_optset,
        "script_invocation": script_invocation,
        "process_passes": process_passes,
        "passes": pass_results,
    }


_CSV_COLUMNS = (
    "nvjitlink_optset",
    "script_invocation",
    "process_pass",
    "cuda_compute_cache_disk",
    "libcudf_jit_lto_in_process",
    "scenario",
    "type_desc",
    "build_ms",
)


_SUMMARY_CSV_COLUMNS = (
    "nvjitlink_optset",
    "scenario",
    "type_desc",
    "cold_ms",
    "warm_ms",
    "warm_pct_of_cold",
    "cold_pct_vs_lto_baseline",
    "warm_pct_vs_lto_baseline",
)


def summarize_bench_csv(raw_path: Path, summary_path: Path) -> None:
    """Read raw benchmark CSV (per-link rows); write a pivot-style summary for analysis.

    Expects ``script_invocation`` 1 = cold disk, 2 = warm disk (as produced by the axis shell).
    Percent columns: ``warm_pct_of_cold`` = 100×warm/cold; ``*_pct_vs_lto_baseline`` = 100×(x/lto−1)
    for the same scenario and invocation class. Baseline rows (``lto``) use 0 for those deltas.
    Appends a **TOTAL** row per ``nvjitlink_optset`` (sum of A–E present in the raw file).
    """
    with raw_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty or missing benchmark CSV: {raw_path}")

    # ms[optset][scenario][inv] = float
    ms: dict[str, dict[str, dict[int, float]]] = {}
    type_desc: dict[tuple[str, str], str] = {}
    opt_order: list[str] = []

    for r in rows:
        opt = r["nvjitlink_optset"]
        scen = r["scenario"]
        inv = int(r["script_invocation"])
        val = float(r["build_ms"])
        if opt not in ms:
            ms[opt] = {}
            opt_order.append(opt)
        ms[opt].setdefault(scen, {})[inv] = val
        type_desc[(opt, scen)] = r.get("type_desc", "")

    known_scen = ["A", "B", "C", "D", "E"]
    all_scen: set[str] = set()
    for d in ms.values():
        all_scen |= set(d.keys())
    scen_order = [s for s in known_scen if s in all_scen] + sorted(
        all_scen - set(known_scen)
    )

    def lto_ms(scen: str, inv: int) -> float | None:
        if "lto" not in ms or scen not in ms["lto"]:
            return None
        return ms["lto"][scen].get(inv)

    def pct_vs_baseline(
        opt: str, scen: str, inv: int, value: float | None
    ) -> str:
        if value is None:
            return ""
        if opt == "lto":
            return "0"
        base = lto_ms(scen, inv)
        if base is None or base <= 0:
            return ""
        return f"{100.0 * (value / base - 1.0):.2f}"

    out_rows: list[dict[str, str]] = []

    for opt in opt_order:
        cold_sum = 0.0
        warm_sum = 0.0
        for scen in scen_order:
            if scen not in ms[opt]:
                continue
            d = ms[opt][scen]
            cold = d.get(1)
            warm = d.get(2)
            if cold is not None:
                cold_sum += cold
            if warm is not None:
                warm_sum += warm

            warm_pct_of_cold = ""
            if cold is not None and warm is not None and cold > 0:
                warm_pct_of_cold = f"{100.0 * warm / cold:.2f}"

            out_rows.append(
                {
                    "nvjitlink_optset": opt,
                    "scenario": scen,
                    "type_desc": type_desc.get((opt, scen), ""),
                    "cold_ms": f"{cold:.6f}" if cold is not None else "",
                    "warm_ms": f"{warm:.6f}" if warm is not None else "",
                    "warm_pct_of_cold": warm_pct_of_cold,
                    "cold_pct_vs_lto_baseline": pct_vs_baseline(opt, scen, 1, cold),
                    "warm_pct_vs_lto_baseline": pct_vs_baseline(opt, scen, 2, warm),
                }
            )

        warm_pct_tot = ""
        if cold_sum > 0 and warm_sum > 0:
            warm_pct_tot = f"{100.0 * warm_sum / cold_sum:.2f}"
        lto_cold_tot = sum(
            ms["lto"][s][1]
            for s in scen_order
            if "lto" in ms and s in ms["lto"] and 1 in ms["lto"][s]
        )
        lto_warm_tot = sum(
            ms["lto"][s][2]
            for s in scen_order
            if "lto" in ms and s in ms["lto"] and 2 in ms["lto"][s]
        )
        cold_pct_tot = ""
        warm_pct_lto_tot = ""
        if opt != "lto" and lto_cold_tot > 0 and cold_sum > 0:
            cold_pct_tot = f"{100.0 * (cold_sum / lto_cold_tot - 1.0):.2f}"
        if opt == "lto":
            cold_pct_tot = "0"
            warm_pct_lto_tot = "0"
        elif lto_warm_tot > 0 and warm_sum > 0:
            warm_pct_lto_tot = f"{100.0 * (warm_sum / lto_warm_tot - 1.0):.2f}"

        out_rows.append(
            {
                "nvjitlink_optset": opt,
                "scenario": "TOTAL",
                "type_desc": "",
                "cold_ms": f"{cold_sum:.6f}" if cold_sum > 0 else "",
                "warm_ms": f"{warm_sum:.6f}" if warm_sum > 0 else "",
                "warm_pct_of_cold": warm_pct_tot,
                "cold_pct_vs_lto_baseline": cold_pct_tot,
                "warm_pct_vs_lto_baseline": warm_pct_lto_tot,
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_CSV_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)


def append_report_to_csv(path: Path, report: dict[str, Any]) -> None:
    """Append one CSV row per link timing event; write header if *path* is missing or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if new_file:
            w.writeheader()
        for p in report["passes"]:
            ax = p["axes_interpretation"]
            for ev in p["link_events"]:
                w.writerow(
                    {
                        "nvjitlink_optset": report["nvjitlink_optset"],
                        "script_invocation": report["script_invocation"],
                        "process_pass": p["process_pass"],
                        "cuda_compute_cache_disk": ax["cuda_compute_cache_disk"],
                        "libcudf_jit_lto_in_process": ax["libcudf_jit_lto_in_process"],
                        "scenario": ev["type_letter"],
                        "type_desc": ev["type_desc"],
                        "build_ms": ev["build_ms"],
                    }
                )


def _print_human_report(report: dict[str, Any]) -> None:
    opt = report["nvjitlink_optset"]
    inv = report["script_invocation"]
    print(
        "AlgorithmPlanner::build() (Murmur JIT+LTO), CUDF_JIT_LTO_LINK_TIMING=1\n"
        f"nvjitlink_optset={opt} script_invocation={inv}\n"
    )
    for p in report["passes"]:
        pn = p["process_pass"]
        ax = p["axes_interpretation"]
        evs = p["link_events"]
        nexp = p["expected_scenarios"]
        print(
            f"--- process_pass={pn} "
            f"(cuda_compute_cache_disk~{ax['cuda_compute_cache_disk']}, "
            f"libcudf_jit_lto_in_process~{ax['libcudf_jit_lto_in_process']}) ---"
        )
        if len(evs) != nexp:
            print(
                f"  Expected {nexp} timing lines, got {len(evs)} "
                "(fragment collapse, wrong build, or cache hit)."
            )
        if not evs:
            print("  (no nvJitLink build lines — expected for pass 2 if in-memory cache hit)\n")
            continue
        for row in evs:
            letter = row["type_letter"]
            desc = row["type_desc"]
            ms = row["build_ms"]
            print(f"  Type {letter} ({desc}): {ms:.6f} ms")
        print("")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summarize-bench-csv",
        type=Path,
        metavar="RAW.csv",
        help="Read raw per-link benchmark CSV and exit after writing summary (see --summary-output)",
    )
    p.add_argument(
        "--summary-output",
        type=Path,
        metavar="PATH",
        help="Summary CSV path (default: RAW stem + _summary.csv next to RAW)",
    )
    p.add_argument(
        "--script-invocation",
        type=int,
        default=1,
        metavar="N",
        help="1 = first python process after clearing ~/.nv/ComputeCache; 2 = second process (warm disk cache). Default: 1",
    )
    p.add_argument(
        "--process-passes",
        type=int,
        default=1,
        metavar="N",
        help="In-process repetitions of the full table sweep (default: 1).",
    )
    p.add_argument(
        "--nvjitlink-optset",
        default="lto",
        metavar="NAME",
        help="Label for this run (e.g. matches CUDF_JIT_LTO_NVJITLINK_OPTIONS preset in the shell driver). Default: lto",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        metavar="PATH",
        help="Write full report as JSON to PATH",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        metavar="PATH",
        help="Append link timing rows to PATH (creates file and header if missing or empty)",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress human-readable report (use with --output-csv or --output-json)",
    )
    args = p.parse_args(argv)

    if args.summarize_bench_csv is not None:
        raw = args.summarize_bench_csv
        if not raw.is_file():
            p.error(f"not a file: {raw}")
        out = args.summary_output
        if out is None:
            out = raw.with_name(f"{raw.stem}_summary.csv")
        summarize_bench_csv(raw, out)
        print(f"Wrote summary CSV: {out}")
        return 0

    if args.process_passes < 1:
        p.error("--process-passes must be >= 1")

    report = run_benchmark(
        script_invocation=args.script_invocation,
        process_passes=args.process_passes,
        nvjitlink_optset=args.nvjitlink_optset,
    )

    if args.output_json:
        args.output_json.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    if args.output_csv:
        append_report_to_csv(args.output_csv, report)

    if not args.quiet:
        _print_human_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
