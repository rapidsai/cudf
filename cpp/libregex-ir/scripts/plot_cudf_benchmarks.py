#!/usr/bin/env python3

# Copyright (c) 2026, Regex IR contributors.
# SPDX-License-Identifier: Apache-2.0

"""Render presentation-ready throughput charts from an NVBench JSON export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import ticker


BACKGROUND = "#0B0B0B"
PANEL = "#151515"
FOREGROUND = "#F5F5F5"
MUTED = "#B8B8B8"
GRID = "#3B3B3B"
NVIDIA_GREEN = "#76B900"
NVIDIA_LIGHT_GREEN = "#A4D65E"
CUDF_GRAY = "#D1D5DB"
FAMILY_ORDER = ("contains", "count", "extract", "replace", "split")
DEFAULT_ROWS = [
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    2_097_152,
    4_194_304,
    8_388_607,
]
README_RESULTS_BEGIN = "<!-- BEGIN GENERATED CUDF API RESULTS -->"
README_RESULTS_END = "<!-- END GENERATED CUDF API RESULTS -->"


@dataclass(frozen=True)
class Measurement:
    """One paired Regex IR/cuDF throughput result."""

    family: str
    case: str
    rows: int
    string_bytes: int
    regex_ir_seconds: float
    cudf_seconds: float

    @property
    def regex_ir_mrows(self) -> float:
        """Return Regex IR throughput in millions of rows per second."""

        return self.rows / self.regex_ir_seconds / 1_000_000.0

    @property
    def cudf_mrows(self) -> float:
        """Return cuDF throughput in millions of rows per second."""

        return self.rows / self.cudf_seconds / 1_000_000.0

    @property
    def speedup(self) -> float:
        """Return Regex IR throughput divided by cuDF throughput."""

        return self.regex_ir_mrows / self.cudf_mrows


def summary_value(state: dict, tag: str) -> float:
    """Read a scalar summary value from an NVBench state."""

    for summary in state["summaries"]:
        if summary["tag"] == tag:
            return float(summary["data"][0]["value"])
    raise ValueError(f"state {state['name']!r} does not contain {tag!r}")


def state_key(name: str) -> str:
    """Remove the device selector from an NVBench state name."""

    return re.sub(r"^Device=\d+\s+", "", name)


def case_label(family: str, name: str) -> str:
    """Create a compact presentation label from an NVBench state name."""

    fields = dict(re.findall(r"([A-Za-z]+)=([^ ]+)", name))
    if family == "extract":
        return f"{fields['Groups']} capture group{'s' if fields['Groups'] != '1' else ''}"
    if family == "replace":
        operation = "backreference" if fields["Type"] == "backref" else "plain"
        return f"pattern {fields['Pattern']} · {operation}"
    return f"pattern {fields['Pattern']}"


def state_fields(name: str) -> dict[str, str]:
    """Parse NVBench axis assignments from a state name."""

    return dict(re.findall(r"([A-Za-z]+)=([^ ]+)", name))


def load_measurements(
    paths: list[Path], rows: list[int], string_bytes: list[int], hit_rate: int
) -> list[Measurement]:
    """Load paired API measurements from one or more NVBench JSON files."""

    benchmarks: dict[str, dict] = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            document = json.load(source)
        for entry in document["benchmarks"]:
            if entry["name"] not in benchmarks:
                benchmarks[entry["name"]] = entry
            else:
                benchmarks[entry["name"]]["states"].extend(entry["states"])
    measurements: list[Measurement] = []
    for family in FAMILY_ORDER:
        regex_name = f"regex_ir/{family}"
        cudf_name = f"cudf/{family}"
        if regex_name not in benchmarks or cudf_name not in benchmarks:
            raise ValueError(f"input is missing {regex_name!r} or {cudf_name!r}")

        def selected(state: dict) -> bool:
            fields = state_fields(state["name"])
            if int(fields["Rows"]) not in rows or int(fields["StringBytes"]) not in string_bytes:
                return False
            return family != "contains" or int(fields["HitRate"]) == hit_rate

        regex_states = {
            state_key(state["name"]): state
            for state in benchmarks[regex_name]["states"]
            if selected(state)
        }
        cudf_states = {
            state_key(state["name"]): state
            for state in benchmarks[cudf_name]["states"]
            if selected(state)
        }
        common_keys = regex_states.keys() & cudf_states.keys()
        if not common_keys:
            raise ValueError(f"Regex IR and cuDF have no common states for {family}")

        for key, regex_state in regex_states.items():
            if key not in common_keys:
                continue
            cudf_state = cudf_states[key]
            regex_rows = int(summary_value(regex_state, "nv/element_count/Rows"))
            cudf_rows = int(summary_value(cudf_state, "nv/element_count/Rows"))
            if regex_rows != cudf_rows:
                raise ValueError(f"row counts differ for {family} {key}")
            measurements.append(
                Measurement(
                    family=family,
                    case=case_label(family, key),
                    rows=regex_rows,
                    string_bytes=int(state_fields(key)["StringBytes"]),
                    regex_ir_seconds=summary_value(
                        regex_state, "nv/cold/time/gpu/mean"
                    ),
                    cudf_seconds=summary_value(cudf_state, "nv/cold/time/gpu/mean"),
                )
            )
        family_measurements = [item for item in measurements if item.family == family]
        expected_geometries = {
            (row_count, width) for row_count in rows for width in string_bytes
        }
        actual_geometries = {
            (item.rows, item.string_bytes) for item in family_measurements
        }
        if actual_geometries != expected_geometries:
            missing = sorted(expected_geometries - actual_geometries)
            raise ValueError(f"{family} is missing requested geometries: {missing}")
        expected_cases = {item.case for item in family_measurements}
        for geometry in sorted(actual_geometries):
            geometry_cases = {
                item.case
                for item in family_measurements
                if (item.rows, item.string_bytes) == geometry
            }
            if geometry_cases != expected_cases:
                raise ValueError(f"{family} has an incomplete case set at {geometry}")
    return measurements


def geometric_mean(values: Iterable[float]) -> float:
    """Return the geometric mean of positive values."""

    materialized = list(values)
    return math.exp(sum(math.log(value) for value in materialized) / len(materialized))


def configure_style() -> None:
    """Apply the dark NVIDIA-inspired presentation theme."""

    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": FOREGROUND,
            "axes.titlecolor": FOREGROUND,
            "xtick.color": MUTED,
            "ytick.color": FOREGROUND,
            "text.color": FOREGROUND,
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 26,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def compact_number(value: int) -> str:
    """Format an axis value compactly without hiding its magnitude."""

    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return str(value)


def geometry_label(rows: list[int], string_bytes: list[int]) -> str:
    """Describe the row and width sweep used by a report."""

    if len(rows) > 4:
        row_values = (
            f"{compact_number(rows[0])}–{compact_number(rows[-1])} "
            f"({len(rows)} points)"
        )
    else:
        row_values = "/".join(compact_number(value) for value in rows)
    width_values = "/".join(str(value) for value in string_bytes)
    return f"rows {row_values}  •  StringBytes {width_values}"


def add_branding(
    fig: plt.Figure, source_name: str, rows: list[int], string_bytes: list[int]
) -> None:
    """Add consistent benchmark context and provenance to a chart."""

    fig.text(
        0.055,
        0.025,
        f"RTX A6000  •  {geometry_label(rows, string_bytes)}  •  warm GPU execution",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.945,
        0.025,
        f"Source: {source_name}",
        color=MUTED,
        fontsize=10,
        horizontalalignment="right",
    )
    fig.text(
        0.945,
        0.955,
        "REGEX IR",
        color=NVIDIA_GREEN,
        fontsize=15,
        fontweight="bold",
        horizontalalignment="right",
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    """Write editable SVG and slide-resolution PNG variants."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), facecolor=BACKGROUND)
    fig.savefig(output_stem.with_suffix(".png"), dpi=160, facecolor=BACKGROUND)
    plt.close(fig)


@dataclass(frozen=True)
class GeometrySummary:
    """Geometric throughput summary for one API and input geometry."""

    rows: int
    string_bytes: int
    cases: int
    regex_ir_mrows: float
    cudf_mrows: float
    speedup: float
    wins: int


def summarize_family(
    measurements: list[Measurement], family: str
) -> list[GeometrySummary]:
    """Aggregate one API's pattern cases independently for each geometry."""

    selected = [item for item in measurements if item.family == family]
    geometries = sorted({(item.rows, item.string_bytes) for item in selected})
    summaries = []
    for rows, string_bytes in geometries:
        values = [
            item
            for item in selected
            if item.rows == rows and item.string_bytes == string_bytes
        ]
        regex_ir_mrows = geometric_mean(item.regex_ir_mrows for item in values)
        cudf_mrows = geometric_mean(item.cudf_mrows for item in values)
        summaries.append(
            GeometrySummary(
                rows=rows,
                string_bytes=string_bytes,
                cases=len(values),
                regex_ir_mrows=regex_ir_mrows,
                cudf_mrows=cudf_mrows,
                speedup=regex_ir_mrows / cudf_mrows,
                wins=sum(item.speedup > 1.0 for item in values),
            )
        )
    return summaries


def api_sweep_chart(
    measurements: list[Measurement],
    family: str,
    output_directory: Path,
    source_name: str,
    rows: list[int],
    string_bytes: list[int],
) -> None:
    """Render one API-specific row-count and string-width scaling chart."""

    summaries = summarize_family(measurements, family)
    widths = sorted({summary.string_bytes for summary in summaries})
    fig, axes = plt.subplots(1, len(widths), figsize=(16, 9), sharey=True)
    if len(widths) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.08, right=0.96, top=0.75, bottom=0.16, wspace=0.12)
    all_throughputs = [
        value
        for summary in summaries
        for value in (summary.regex_ir_mrows, summary.cudf_mrows)
    ]
    y_min = min(all_throughputs) / 1.7
    y_max = max(all_throughputs) * 1.8

    for axis, width in zip(axes, widths):
        values = [summary for summary in summaries if summary.string_bytes == width]
        values.sort(key=lambda item: item.rows)
        x_values = [summary.rows for summary in values]
        regex_values = [summary.regex_ir_mrows for summary in values]
        cudf_values = [summary.cudf_mrows for summary in values]
        axis.plot(
            x_values,
            regex_values,
            color=NVIDIA_GREEN,
            marker="o",
            markersize=9,
            linewidth=3,
            label="Regex IR",
            zorder=3,
        )
        axis.plot(
            x_values,
            cudf_values,
            color=CUDF_GRAY,
            marker="o",
            markersize=7,
            linewidth=2.2,
            label="cuDF",
            zorder=2,
        )
        annotation_indices = {0, len(values) // 2, len(values) - 1}
        for index, summary in enumerate(values):
            if index not in annotation_indices:
                continue
            axis.annotate(
                f"{summary.speedup:.1f}×",
                (summary.rows, summary.regex_ir_mrows),
                xytext=(0, 12),
                textcoords="offset points",
                color=NVIDIA_LIGHT_GREEN,
                fontsize=11,
                fontweight="bold",
                horizontalalignment="center",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(y_min, y_max)
        tick_indices = {
            round(index * (len(x_values) - 1) / 4) for index in range(5)
        }
        tick_values = [
            value for index, value in enumerate(x_values) if index in tick_indices
        ]
        axis.set_xticks(
            tick_values, [compact_number(value) for value in tick_values]
        )
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.grid(color=GRID, linewidth=0.8, alpha=0.75, zorder=0)
        axis.set_title(f"StringBytes = {width}", pad=16, fontsize=18)
        axis.set_xlabel("Rows")
    axes[0].set_ylabel("Throughput (million rows / second, log scale)")
    axes[-1].legend(loc="lower right", fontsize=13)

    case_count = max(summary.cases for summary in summaries)
    case_word = "case" if case_count == 1 else "cases"
    fig.text(
        0.055,
        0.91,
        f"{family.title()} throughput by row count and width parameter",
        fontsize=29,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.845,
        f"Geometric mean across {case_count} {family} {case_word} at each geometry • selected labels show speedup • higher is better",
        color=MUTED,
        fontsize=14,
    )
    add_branding(fig, source_name, rows, string_bytes)
    save_figure(fig, output_directory / f"cudf-api-{family}-throughput-sweep")


def export_csv(measurements: list[Measurement], path: Path) -> None:
    """Write the plotted values in a compact, reviewable form."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "family",
                "case",
                "rows",
                "string_bytes",
                "regex_ir_gpu_ms",
                "cudf_gpu_ms",
                "regex_ir_mrows_per_second",
                "cudf_mrows_per_second",
                "speedup",
            ]
        )
        for measurement in measurements:
            writer.writerow(
                [
                    measurement.family,
                    measurement.case,
                    measurement.rows,
                    measurement.string_bytes,
                    f"{measurement.regex_ir_seconds * 1000:.9f}",
                    f"{measurement.cudf_seconds * 1000:.9f}",
                    f"{measurement.regex_ir_mrows:.6f}",
                    f"{measurement.cudf_mrows:.6f}",
                    f"{measurement.speedup:.6f}",
                ]
            )


def joined_names(names: list[str]) -> str:
    """Join presentation labels into a readable English list."""

    if len(names) < 2:
        return "".join(names)
    if len(names) == 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def write_readme_results(measurements: list[Measurement], path: Path) -> None:
    """Replace the generated README benchmark block with current results."""

    lines = [
        README_RESULTS_BEGIN,
        "",
        "| API | Expressions | Geometries | Paired measurements | Regex IR geometric throughput (M rows/s) | cuDF geometric throughput (M rows/s) | Geometric speedup | Pair wins |",
        "|:---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for family in FAMILY_ORDER:
        values = [item for item in measurements if item.family == family]
        cases = len({item.case for item in values})
        geometries = len({(item.rows, item.string_bytes) for item in values})
        regex_throughput = geometric_mean(item.regex_ir_mrows for item in values)
        cudf_throughput = geometric_mean(item.cudf_mrows for item in values)
        wins = sum(item.speedup > 1.0 for item in values)
        lines.append(
            f"| {family} | {cases} | {geometries} | {len(values)} | "
            f"{regex_throughput:,.3f} | {cudf_throughput:,.3f} | "
            f"{regex_throughput / cudf_throughput:.3f}x | "
            f"{wins}–{len(values) - wins} |"
        )

    regex_throughput = geometric_mean(item.regex_ir_mrows for item in measurements)
    cudf_throughput = geometric_mean(item.cudf_mrows for item in measurements)
    wins = sum(item.speedup > 1.0 for item in measurements)
    case_count = sum(
        len({item.case for item in measurements if item.family == family})
        for family in FAMILY_ORDER
    )
    geometry_count = len({(item.rows, item.string_bytes) for item in measurements})
    lines.extend(
        [
            f"| **All APIs** | **{case_count}** | **{geometry_count}** | "
            f"**{len(measurements)}** | **{regex_throughput:,.3f}** | "
            f"**{cudf_throughput:,.3f}** | "
            f"**{regex_throughput / cudf_throughput:.3f}x** | "
            f"**{wins}–{len(measurements) - wins}** |",
            "",
        ]
    )

    perfect_families = [
        family
        for family in FAMILY_ORDER
        if all(item.speedup > 1.0 for item in measurements if item.family == family)
    ]
    loss_families = [family for family in FAMILY_ORDER if family not in perfect_families]
    narrowest = min(measurements, key=lambda item: item.speedup)
    largest = max(measurements, key=lambda item: item.speedup)
    if loss_families:
        win_summary = (
            f"{joined_names(perfect_families).capitalize()} won every state; losses are confined "
            f"to {joined_names(loss_families)} states and remain visible below."
        )
    else:
        win_summary = f"{joined_names(perfect_families).capitalize()} won every state."
    result_summary = (
        f"Regex IR won {wins} of {len(measurements)} paired measurements. "
        f"{win_summary} The narrowest "
        f"result was {narrowest.family} {narrowest.case} at {narrowest.rows:,} rows/"
        f"{narrowest.string_bytes} bytes ({narrowest.speedup:.3f}x), while the largest was "
        f"{largest.family} {largest.case} at {largest.rows:,} rows/"
        f"{largest.string_bytes} bytes ({largest.speedup:.3f}x). The results describe this "
        "hardware, corpus, and allocation contract—not every regex or deployment. Use "
        "the commands above to reproduce the matrix."
    )
    lines.extend(
        textwrap.wrap(result_summary, width=88)
        + [
            "",
            "The tables below report geometric mean throughput across each API's expression",
            "cases independently at every row-count/`StringBytes` geometry.",
            "",
        ]
    )

    for family in FAMILY_ORDER:
        lines.extend(
            [
                f"#### {family.title()}",
                "",
                f"[PNG chart](docs/_static/benchmarks/cudf-api-{family}-throughput-sweep.png) ·",
                f"[SVG chart](docs/_static/benchmarks/cudf-api-{family}-throughput-sweep.svg) ·",
                f"[case-level CSV](docs/_static/benchmarks/cudf-api-{family}-throughput-data.csv)",
                "",
                "| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |",
                "|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for summary in summarize_family(measurements, family):
            lines.append(
                f"| {summary.rows:,} | {summary.string_bytes} | {summary.cases} | "
                f"{summary.regex_ir_mrows:,.3f} | {summary.cudf_mrows:,.3f} | "
                f"{summary.speedup:.3f}x | "
                f"{summary.wins}–{summary.cases - summary.wins} |"
            )
        lines.append("")
    lines.extend(
        [
            "[All API measurements](docs/_static/benchmarks/cudf-api-throughput-data.csv)",
            "",
            README_RESULTS_END,
        ]
    )

    readme = path.read_text(encoding="utf-8")
    begin = readme.index(README_RESULTS_BEGIN)
    end = readme.index(README_RESULTS_END, begin) + len(README_RESULTS_END)
    path.write_text(readme[:begin] + "\n".join(lines) + readme[end:], encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated positive integer axis list."""

    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("axis values must be integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("axis values must be positive")
    return sorted(set(values))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Render NVIDIA-styled Regex IR/cuDF throughput charts from NVBench JSON."
    )
    parser.add_argument(
        "nvbench_json",
        type=Path,
        nargs="+",
        help="one or more NVBench JSON exports to plot",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("docs/_static/benchmarks"),
        help="destination for SVG, PNG, and compact CSV files",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        help="README whose generated cuDF API result block should be refreshed",
    )
    parser.add_argument(
        "--rows",
        type=parse_int_list,
        default=DEFAULT_ROWS,
        help="comma-separated row axes to plot",
    )
    parser.add_argument(
        "--string-bytes",
        type=parse_int_list,
        default=[64, 128, 256],
        help="comma-separated StringBytes axes to plot",
    )
    parser.add_argument(
        "--hit-rate", type=int, default=50, help="contains HitRate axis to plot"
    )
    return parser.parse_args()


def main() -> None:
    """Generate one parameterized line chart and report per regex API."""

    arguments = parse_arguments()
    measurements = load_measurements(
        arguments.nvbench_json,
        arguments.rows,
        arguments.string_bytes,
        arguments.hit_rate,
    )
    configure_style()
    source_name = "NVBench mean GPU time • values archived as CSV"
    for family in FAMILY_ORDER:
        api_sweep_chart(
            measurements,
            family,
            arguments.output_directory,
            source_name,
            arguments.rows,
            arguments.string_bytes,
        )
        export_csv(
            [item for item in measurements if item.family == family],
            arguments.output_directory / f"cudf-api-{family}-throughput-data.csv",
        )
    export_csv(measurements, arguments.output_directory / "cudf-api-throughput-data.csv")
    if arguments.readme is not None:
        write_readme_results(measurements, arguments.readme)


if __name__ == "__main__":
    main()
