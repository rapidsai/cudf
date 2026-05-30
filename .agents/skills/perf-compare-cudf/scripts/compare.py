#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Compare cudf nvbench CSV results between a `main` run and a `pr` run.

Rows are matched by (benchmark name, axis values). For each matched config we
compute the GPU-time delta (PR - main) / main and emit a markdown report. A
config is flagged "significant" when |delta| >= threshold AND |delta| exceeds
the nvbench noise on either side, so noise-dominated swings don't get flagged.

Each input directory holds one CSV per benchmark binary, named after the binary
(as produced by `<BENCH> --csv <dir>/<BENCH>.csv`). Only CSVs present in BOTH
directories (matched by filename) are compared.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

# nvbench CSV column names we read directly.
COL_BENCHMARK = "Benchmark"
COL_GPU_TIME = "GPU Time (sec)"
COL_NOISE = "Noise"
COL_SKIPPED = "Skipped"

# Columns that are never benchmark axes (identity + the metrics we don't axis on).
NON_AXIS = {
    COL_BENCHMARK,
    "Device",
    "Device Name",
    COL_SKIPPED,
    "Samples",
    "CPU Time (sec)",
    COL_NOISE,
    COL_GPU_TIME,
}

# nvbench emits a different set of metric/output columns per benchmark, so we
# can't enumerate them. Instead we drop any column whose name looks like a
# measured output; everything else is treated as an axis.
METRIC_HINTS = (
    "per_second",
    "per_sec",
    "bytes_per_second",
    "BW",
    "Noise",
    "Samples",
    "Time (sec)",
    "memory",
    "encoded",
    "Util",
    "size (bytes)",
    "throughput",
)


def looks_like_metric(column: str) -> bool:
    return any(hint.lower() in column.lower() for hint in METRIC_HINTS)


def pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def axes_str(axes: dict[str, str]) -> str:
    return " ".join(
        f"{name}={value}" for name, value in axes.items() if value != ""
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored markdown table from headers and string cells."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    return "\n".join(lines) + "\n"


@dataclass
class Comparison:
    """One benchmark config measured on both branches."""

    bench: str
    axes: dict[str, str]
    main_ms: float
    pr_ms: float
    delta: float
    main_noise: float
    pr_noise: float

    def significant(self, threshold: float) -> bool:
        return abs(self.delta) >= threshold and abs(self.delta) > max(
            self.main_noise, self.pr_noise
        )

    def flag(self, threshold: float) -> str:
        if not self.significant(threshold):
            return ""
        return "**FASTER**" if self.delta < 0 else "**SLOWER**"

    def noise_str(self) -> str:
        return f"{self.main_noise * 100:.1f}%/{self.pr_noise * 100:.1f}%"


@dataclass
class SuiteResult:
    """Comparison outcome for a single benchmark binary."""

    name: str
    rows: list[Comparison]
    only_pr: list  # config keys present only in the PR run
    only_main: list  # config keys present only in the main run


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def config_key(row: dict):
    """Identity of a benchmark config: its name plus all (axis, value) pairs."""
    axes = sorted(
        c for c in row if c not in NON_AXIS and not looks_like_metric(c)
    )
    return (row[COL_BENCHMARK],) + tuple((axis, row[axis]) for axis in axes)


def compare_suite(
    name: str, pr_dir: Path, main_dir: Path
) -> SuiteResult | None:
    pr_csv, main_csv = pr_dir / f"{name}.csv", main_dir / f"{name}.csv"
    if not pr_csv.exists() or not main_csv.exists():
        return None

    pr_by_key = {config_key(row): row for row in read_rows(pr_csv)}
    main_by_key = {config_key(row): row for row in read_rows(main_csv)}

    rows: list[Comparison] = []
    for key in sorted(pr_by_key.keys() & main_by_key.keys()):
        pr_row, main_row = pr_by_key[key], main_by_key[key]
        if (
            pr_row.get(COL_SKIPPED) == "Yes"
            or main_row.get(COL_SKIPPED) == "Yes"
        ):
            continue
        try:
            pr_time = float(pr_row[COL_GPU_TIME])
            main_time = float(main_row[COL_GPU_TIME])
        except (KeyError, ValueError):
            continue
        if main_time <= 0:
            continue
        rows.append(
            Comparison(
                bench=key[0],
                axes={axis: value for axis, value in key[1:]},
                main_ms=main_time * 1000,
                pr_ms=pr_time * 1000,
                delta=(pr_time - main_time) / main_time,
                main_noise=float(main_row.get(COL_NOISE) or 0),
                pr_noise=float(pr_row.get(COL_NOISE) or 0),
            )
        )

    return SuiteResult(
        name=name,
        rows=rows,
        only_pr=sorted(pr_by_key.keys() - main_by_key.keys()),
        only_main=sorted(main_by_key.keys() - pr_by_key.keys()),
    )


def render_header(threshold: float) -> str:
    return (
        "# Benchmark Comparison: main vs PR\n\n"
        "- GPU Time in ms. Δ = (PR - main) / main. Negative = PR faster.\n"
        f"- **Significant**: |Δ| >= {threshold * 100:.0f}% AND larger than max(noise) of either side.\n"
        "- Fill in hardware (GPU/driver/CUDA), branch SHAs, and axis coverage manually.\n\n"
    )


def render_summary(suites: list[SuiteResult], threshold: float) -> str:
    rows = [
        [
            suite.name,
            str(len(suite.rows)),
            str(sum(1 for r in suite.rows if r.significant(threshold))),
        ]
        for suite in suites
    ]
    return (
        "## Summary\n\n"
        + markdown_table(
            ["Benchmark Suite", "# Configs", "# Significant"], rows
        )
        + "\n"
    )


def render_top(suites: list[SuiteResult], top: int) -> str:
    everything = [(suite.name, row) for suite in suites for row in suite.rows]
    everything.sort(key=lambda pair: abs(pair[1].delta), reverse=True)
    rows = [
        [
            f"{suite_name}/{r.bench}",
            axes_str(r.axes),
            f"{r.main_ms:.3f}",
            f"{r.pr_ms:.3f}",
            pct(r.delta),
            r.noise_str(),
        ]
        for suite_name, r in everything[:top]
    ]
    return (
        f"## Top {top} by |Δ|\n\n"
        + markdown_table(
            [
                "Suite / bench",
                "axes",
                "main (ms)",
                "PR (ms)",
                "Δ",
                "noise(m/p)",
            ],
            rows,
        )
        + "\n"
    )


def render_suite(suite: SuiteResult, threshold: float) -> str:
    out = [f"## {suite.name}\n\n"]
    if suite.only_pr or suite.only_main:
        out.append(
            f"_only-on-PR configs: {len(suite.only_pr)}, "
            f"only-on-main: {len(suite.only_main)}_\n\n"
        )

    by_bench: dict[str, list[Comparison]] = {}
    for row in suite.rows:
        by_bench.setdefault(row.bench, []).append(row)

    for bench, rows in by_bench.items():
        axis_keys = sorted({axis for r in rows for axis in r.axes})
        headers = axis_keys + [
            "main (ms)",
            "PR (ms)",
            "Δ",
            "noise(m/p)",
            "flag",
        ]
        table_rows = [
            [str(r.axes.get(axis, "")) for axis in axis_keys]
            + [
                f"{r.main_ms:.3f}",
                f"{r.pr_ms:.3f}",
                pct(r.delta),
                f"{r.main_noise * 100:.2f}%/{r.pr_noise * 100:.2f}%",
                r.flag(threshold),
            ]
            for r in sorted(
                rows,
                key=lambda r: tuple(str(r.axes.get(a, "")) for a in axis_keys),
            )
        ]
        out.append(f"### `{bench}`\n\n")
        out.append(markdown_table(headers, table_rows))
        out.append("\n")
    return "".join(out)


def render_report(
    suites: list[SuiteResult], threshold: float, top: int
) -> str:
    parts = [
        render_header(threshold),
        render_summary(suites, threshold),
        render_top(suites, top),
    ]
    parts += [render_suite(suite, threshold) for suite in suites]
    return "".join(parts)


def print_significant(suites: list[SuiteResult], threshold: float) -> None:
    print("\n=== Significant differences ===")
    found = False
    for suite in suites:
        for r in suite.rows:
            if r.significant(threshold):
                found = True
                tag = "FASTER" if r.delta < 0 else "SLOWER"
                print(
                    f"  [{tag}] {suite.name}/{r.bench} {axes_str(r.axes)} "
                    f"main={r.main_ms:.3f}ms PR={r.pr_ms:.3f}ms d={pct(r.delta)}"
                )
    if not found:
        print("  (none — all within noise / below threshold)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pr", required=True, type=Path, help="dir with PR-branch CSVs"
    )
    parser.add_argument(
        "--main", required=True, type=Path, help="dir with main-branch CSVs"
    )
    parser.add_argument(
        "--report", required=True, type=Path, help="output markdown path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="significance threshold (default 0.05)",
    )
    parser.add_argument(
        "--top", type=int, default=15, help="rows in Top-N table (default 15)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    suite_names = sorted(p.stem for p in args.pr.glob("*.csv"))
    if not suite_names:
        raise SystemExit(f"no CSVs found in {args.pr}")

    suites = [
        s
        for name in suite_names
        if (s := compare_suite(name, args.pr, args.main)) is not None
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(suites, args.threshold, args.top))
    print(f"wrote {args.report}")

    print_significant(suites, args.threshold)


if __name__ == "__main__":
    main()
