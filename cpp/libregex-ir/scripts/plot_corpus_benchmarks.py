#!/usr/bin/env python3
"""Render complete-corpus Regex IR/cuDF case charts and README tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch

from plot_cudf_benchmarks import (
    CUDF_GRAY,
    GRID,
    MUTED,
    NVIDIA_GREEN,
    configure_style,
    geometric_mean,
    save_figure,
    summary_value,
)


SUITE_ORDER = ("openresty", "leipzig", "boost", "mariomka")
SUITE_LABELS = {
    "openresty": "OpenResty",
    "leipzig": "Rust Leipzig",
    "boost": "Boost/GCC",
    "mariomka": "mariomka",
}
SUITE_PANELS = {
    "openresty": (
        ("Generated corpora", 1, 16),
        ("Complete mtent12", 17, 31),
    ),
    "leipzig": (("Complete 3200.txt", 1, 18),),
    "boost": (
        ("Long Twain", 1, 6),
        ("Medium Twain", 7, 12),
        ("Complete C++", 13, 16),
        ("Complete HTML", 17, 22),
    ),
    "mariomka": (("Complete input-text.txt", 1, 3),),
}
SUITE_CASE_NAMES = {
    "openresty": (
        "01_literal_miss",
        "02_short_alt_miss",
        "03_suffix_alt_miss",
        "04_suffix_alt_prose",
        "05_wide_class_miss",
        "06_split_class_miss",
        "07_split_class_prose",
        "08_large_alt_prose",
        "09_large_alt_miss",
        "10_nested_alt",
        "11_long_nested_alt_miss",
        "12_capture_chain_miss",
        "13_capture_chain_random_miss",
        "14_lazy_class_repeat",
        "15_lazy_dot_repeat",
        "16_greedy_dot_repeat",
        "17_anchored_literal",
        "18_literal_prose",
        "19_folded_literal",
        "20_class_suffix",
        "21_name_alternation",
        "22_word_boundary",
        "23_negated_bounded",
        "24_name_literals",
        "25_folded_names",
        "26_short_prefix_names",
        "27_required_prefix_names",
        "28_word_suffix",
        "29_bounded_word_suffix",
        "30_captured_name_suffix",
        "31_quoted_sentence",
    ),
    "leipzig": (
        "01_twain",
        "02_twain_ignore_case",
        "03_shing",
        "04_huck_saw",
        "05_word_nn",
        "06_negated_bounded",
        "07_names",
        "08_names_ignore_case",
        "09_optional_prefix",
        "10_required_prefix",
        "11_tom_river",
        "12_word_ing",
        "13_bounded_ing",
        "14_name_suffix",
        "15_quoted_sentence",
        "16_unicode_symbols",
        "17_math_symbol_property",
        "18_csv_field",
    ),
    "boost": (
        "01_long_twain",
        "02_long_huck",
        "03_long_ing",
        "04_long_line_twain",
        "05_long_names",
        "06_long_names_near_river",
        "07_medium_twain",
        "08_medium_huck",
        "09_medium_ing",
        "10_medium_line_twain",
        "11_medium_names",
        "12_medium_names_near_river",
        "13_cpp_declaration",
        "14_cpp_tokens",
        "15_cpp_include",
        "16_boost_include",
        "17_html_names",
        "18_html_paragraph",
        "19_html_anchor",
        "20_html_heading",
        "21_html_image",
        "22_html_font",
    ),
    "mariomka": ("01_email", "02_uri", "03_ipv4"),
}
README_RESULTS_BEGIN = "<!-- BEGIN GENERATED CORPUS SWEEP RESULTS -->"
README_RESULTS_END = "<!-- END GENERATED CORPUS SWEEP RESULTS -->"
LOG_SCALE_THRESHOLD = 50.0


@dataclass(frozen=True)
class CorpusMeasurement:
    """One paired full-corpus benchmark case."""

    suite: str
    case: int
    case_name: str
    rows: int
    columns: int
    max_string_bytes: int
    input_bytes: int
    regex_ir_seconds: float
    cudf_seconds: float
    regex_ir_jit_ready_seconds: float
    cudf_program_create_seconds: float

    @property
    def regex_ir_gib_per_second(self) -> float:
        """Return Regex IR input throughput in GiB/s."""

        return self.input_bytes / self.regex_ir_seconds / (1024**3)

    @property
    def cudf_gib_per_second(self) -> float:
        """Return cuDF input throughput in GiB/s."""

        return self.input_bytes / self.cudf_seconds / (1024**3)

    @property
    def speedup(self) -> float:
        """Return Regex IR throughput divided by cuDF throughput."""

        return self.cudf_seconds / self.regex_ir_seconds


def selected_geometry(suite: str, case: int) -> tuple[int, int, int]:
    """Return the packed geometry that retains one complete source corpus."""

    if suite == "boost" and case >= 7:
        return (1024, 1, 64)
    return (32768, 8, 256)


def state_key(name: str) -> str:
    """Remove the device selector from an NVBench state name."""

    return re.sub(r"^Device=\d+\s+", "", name)


def state_fields(name: str) -> dict[str, str]:
    """Parse NVBench axis assignments from a state name."""

    return dict(re.findall(r"([A-Za-z]+)=([^ ]+)", name))


def completed(state: dict) -> bool:
    """Return whether an NVBench state contains a measured GPU duration."""

    return any(
        summary.get("tag") == "nv/cold/time/gpu/mean"
        for summary in state.get("summaries", [])
    )


def optional_summary_value(state: dict, *tags: str) -> float:
    """Return the first available summary value, or NaN for an older export."""

    for tag in tags:
        try:
            return summary_value(state, tag)
        except ValueError:
            continue
    return math.nan


def load_measurements(paths: list[Path]) -> list[CorpusMeasurement]:
    """Load the paired packed geometry for every full-corpus expression."""

    benchmarks: dict[str, dict[str, dict]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            document = json.load(source)
        for benchmark in document["benchmarks"]:
            states = benchmarks.setdefault(benchmark["name"], {})
            for state in benchmark["states"]:
                states[state_key(state["name"])] = state

    measurements = []
    for suite in SUITE_ORDER:
        regex_name = f"regex_ir/{suite}"
        cudf_name = f"cudf/{suite}"
        if regex_name not in benchmarks or cudf_name not in benchmarks:
            raise ValueError(f"input is missing {regex_name!r} or {cudf_name!r}")

        for case, case_name in enumerate(SUITE_CASE_NAMES[suite], start=1):
            rows, columns, max_string_bytes = selected_geometry(suite, case)

            def find_state(name: str) -> dict:
                for state in benchmarks[name].values():
                    if not completed(state):
                        continue
                    fields = state_fields(state["name"])
                    if (
                        int(fields["Case"]) == case
                        and int(fields["Rows"]) == rows
                        and int(fields["Columns"]) == columns
                        and int(fields["MaxStringBytes"]) == max_string_bytes
                    ):
                        return state
                raise ValueError(
                    f"{name} is missing case {case} at "
                    f"Rows={rows} Columns={columns} MaxStringBytes={max_string_bytes}"
                )

            regex_state = find_state(regex_name)
            cudf_state = find_state(cudf_name)
            regex_bytes = int(summary_value(regex_state, "nv/gmem/reads/InputBytes"))
            cudf_bytes = int(summary_value(cudf_state, "nv/gmem/reads/InputBytes"))
            if regex_bytes != cudf_bytes:
                raise ValueError(f"input byte counts differ for {suite} case {case}")
            measurements.append(
                CorpusMeasurement(
                    suite=suite,
                    case=case,
                    case_name=case_name,
                    rows=rows,
                    columns=columns,
                    max_string_bytes=max_string_bytes,
                    input_bytes=regex_bytes,
                    regex_ir_seconds=summary_value(
                        regex_state, "nv/cold/time/gpu/mean"
                    ),
                    cudf_seconds=summary_value(cudf_state, "nv/cold/time/gpu/mean"),
                    regex_ir_jit_ready_seconds=optional_summary_value(
                        regex_state,
                        "regex_ir/jit_ready_time",
                        "regex_ir/corpus/compile_time",
                    ),
                    cudf_program_create_seconds=optional_summary_value(
                        cudf_state,
                        "cudf/program_create_time",
                        "regex_ir/corpus/compile_time",
                    ),
                )
            )
    return measurements


def add_branding(figure: plt.Figure, scale_note: str) -> None:
    """Add complete-corpus geometry and provenance to a chart."""

    figure.text(
        0.055,
        0.025,
        f"RTX A6000  •  full corpus  •  warm GPU  •  {scale_note}",
        color=MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.945,
        0.025,
        "NVBench mean GPU time",
        color=MUTED,
        fontsize=10,
        horizontalalignment="right",
    )
    figure.text(
        0.945,
        0.955,
        "REGEX IR",
        color=NVIDIA_GREEN,
        fontsize=15,
        fontweight="bold",
        horizontalalignment="right",
    )


def corpus_case_bar_chart(
    measurements: list[CorpusMeasurement], suite: str, output_directory: Path
) -> None:
    """Render grouped linear bars for one suite's categorical corpus cases."""

    values = sorted(
        (item for item in measurements if item.suite == suite),
        key=lambda item: item.case,
    )
    panel_specs = SUITE_PANELS[suite]
    panel_count = len(panel_specs)
    row_count = 2 if panel_count == 4 else 1
    column_count = 2 if panel_count in (2, 4) else 1
    figure, axes = plt.subplots(row_count, column_count, figsize=(16, 9))
    if panel_count == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)
    figure.subplots_adjust(
        left=0.16 if panel_count == 1 else 0.17,
        right=0.96,
        top=0.80,
        bottom=0.14,
        wspace=0.55,
        hspace=0.52,
    )

    log_panel_count = 0
    for axis, (panel_name, first_case, last_case) in zip(axes, panel_specs):
        panel = [
            item for item in values if first_case <= item.case <= last_case
        ]
        y_values = list(range(len(panel)))
        regex_values = [item.regex_ir_gib_per_second for item in panel]
        cudf_values = [item.cudf_gib_per_second for item in panel]
        panel_min = min(regex_values + cudf_values)
        panel_max = max(regex_values + cudf_values)
        use_log_scale = panel_max / panel_min >= LOG_SCALE_THRESHOLD
        log_panel_count += int(use_log_scale)
        bar_origin = panel_min / 1.4 if use_log_scale else 0.0
        bar_height = 0.32
        baseline_y = [value - bar_height / 2 for value in y_values]
        regex_y = [value + bar_height / 2 for value in y_values]
        axis.barh(
            baseline_y,
            [value - bar_origin for value in cudf_values],
            left=bar_origin,
            height=bar_height,
            color=CUDF_GRAY,
            edgecolor="none",
            zorder=2,
        )
        axis.barh(
            regex_y,
            [value - bar_origin for value in regex_values],
            left=bar_origin,
            height=bar_height,
            color=NVIDIA_GREEN,
            edgecolor="none",
            zorder=3,
        )
        speedup_x = panel_max * (1.28 if use_log_scale else 1.11)
        for y_value, item in zip(y_values, panel):
            axis.text(
                speedup_x,
                y_value,
                f"{item.speedup:.1f}×",
                color=NVIDIA_GREEN,
                fontsize=8.5,
                fontweight="bold",
                verticalalignment="center",
                horizontalalignment="center",
            )
        if use_log_scale:
            axis.set_xscale("log")
            axis.set_xlim(bar_origin, panel_max * 1.85)
            axis.xaxis.set_minor_formatter(ticker.NullFormatter())
            axis.text(
                0.015,
                0.975,
                "LOG",
                transform=axis.transAxes,
                color=MUTED,
                fontsize=8.5,
                fontweight="bold",
                horizontalalignment="left",
                verticalalignment="top",
            )
        else:
            axis.set_xlim(0.0, panel_max * 1.24)
            candidate_ticks = ticker.MaxNLocator(nbins=4, min_n_ticks=3).tick_values(
                0.0, panel_max
            )
            axis.set_xticks(
                [value for value in candidate_ticks if 0.0 <= value <= panel_max]
            )
        axis.set_yticks(
            y_values,
            [
                f"{item.case:02d}  {display_case_name(item.case_name)}"
                for item in panel
            ],
            fontsize=9.5,
        )
        axis.invert_yaxis()
        axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.75, zorder=0)
        axis.set_title(panel_name, pad=10, fontsize=13)

    figure.supxlabel("Input throughput (GiB / second)", y=0.075, fontsize=15)

    figure.legend(
        handles=[
            Patch(facecolor=CUDF_GRAY, label="cuDF baseline"),
            Patch(facecolor=NVIDIA_GREEN, label="Regex IR"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.95, 0.88),
        ncol=2,
        fontsize=12,
        frameon=False,
    )

    figure.text(
        0.055,
        0.91,
        f"{SUITE_LABELS[suite]}: full-corpus throughput",
        fontsize=29,
        fontweight="bold",
    )
    if log_panel_count == panel_count:
        scale_note = "log scales"
    elif log_panel_count == 0:
        scale_note = "linear scale" if panel_count == 1 else "independent linear scales"
    else:
        scale_note = "mixed scales  •  LOG panels marked"
    add_branding(figure, scale_note)
    save_figure(figure, output_directory / f"corpus-{suite}-throughput-cases")


def export_csv(measurements: list[CorpusMeasurement], path: Path) -> None:
    """Write every plotted corpus case and cold setup measurement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "suite",
                "case",
                "case_name",
                "rows",
                "columns",
                "max_string_bytes",
                "input_bytes",
                "regex_ir_gpu_ms",
                "cudf_gpu_ms",
                "regex_ir_gib_per_second",
                "cudf_gib_per_second",
                "speedup",
                "regex_ir_jit_ready_ms",
                "cudf_program_create_ms",
            ]
        )
        for item in measurements:
            writer.writerow(
                [
                    item.suite,
                    item.case,
                    item.case_name,
                    item.rows,
                    item.columns,
                    item.max_string_bytes,
                    item.input_bytes,
                    f"{item.regex_ir_seconds * 1000:.9f}",
                    f"{item.cudf_seconds * 1000:.9f}",
                    f"{item.regex_ir_gib_per_second:.6f}",
                    f"{item.cudf_gib_per_second:.6f}",
                    f"{item.speedup:.6f}",
                    f"{item.regex_ir_jit_ready_seconds * 1000:.6f}",
                    f"{item.cudf_program_create_seconds * 1000:.6f}",
                ]
            )


def display_case_name(name: str) -> str:
    """Turn a stable source-case identifier into a compact table label."""

    return name.split("_", 1)[1].replace("_", " ")


def write_readme_results(measurements: list[CorpusMeasurement], path: Path) -> None:
    """Replace the generated complete-corpus README result block."""

    all_wins = sum(item.speedup > 1.0 for item in measurements)
    all_regex = geometric_mean(
        item.regex_ir_gib_per_second for item in measurements
    )
    all_cudf = geometric_mean(item.cudf_gib_per_second for item in measurements)
    lines = [
        README_RESULTS_BEGIN,
        "",
        "#### Complete-corpus case charts",
        "",
        "The linked presentation charts use uncluttered grouped horizontal bars for every",
        "imported suite without embedding images in this README. Each point is one upstream",
        "regex case over its complete source corpus; throughput uses input bytes rather than",
        "row count so differently packed cuDF columns remain comparable. The 14 Boost scalar",
        "records remain in the suite summary above but are intentionally absent here because",
        "they are repeated literals rather than source corpora.",
        "Panels spanning at least 50x use a marked logarithmic throughput axis; narrower",
        "panels retain a linear axis.",
        "",
        "These cases were rerun on 2026-07-07 with at least five samples, 0.05 seconds of",
        "measured GPU time, a 2% target-noise threshold, and a 10-second per-state timeout.",
        "All 148 engine states completed without warnings, skips, or timeouts, and every",
        "pre-timing Regex IR/cuDF output comparison passed.",
        "JIT-ready time is uncached and spans the regex string through loaded module and",
        "resolved kernel function; corpus setup and the first launch are excluded.",
        "",
        "| Source suite | Corpus expressions | Regex IR geometric throughput (GiB/s) | cuDF geometric throughput (GiB/s) | Geometric speedup | Pair wins | Regex IR JIT-ready mean (ms) | cuDF program-create mean (ms) |",
        "|:---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for suite in SUITE_ORDER:
        values = [item for item in measurements if item.suite == suite]
        regex_throughput = geometric_mean(
            item.regex_ir_gib_per_second for item in values
        )
        cudf_throughput = geometric_mean(
            item.cudf_gib_per_second for item in values
        )
        wins = sum(item.speedup > 1.0 for item in values)
        regex_compile = sum(item.regex_ir_jit_ready_seconds for item in values) / len(
            values
        )
        cudf_compile = sum(item.cudf_program_create_seconds for item in values) / len(
            values
        )
        lines.append(
            f"| {SUITE_LABELS[suite]} | {len(values)} | {regex_throughput:.3f} | "
            f"{cudf_throughput:.3f} | {regex_throughput / cudf_throughput:.3f}x | "
            f"{wins}–{len(values) - wins} | {regex_compile * 1000:.3f} | "
            f"{cudf_compile * 1000:.4f} |"
        )
    all_regex_compile = sum(
        item.regex_ir_jit_ready_seconds for item in measurements
    ) / len(measurements)
    all_cudf_compile = sum(
        item.cudf_program_create_seconds for item in measurements
    ) / len(measurements)
    lines.extend(
        [
            f"| **All complete-corpus suites** | **{len(measurements)}** | **{all_regex:.3f}** | **{all_cudf:.3f}** | **{all_regex / all_cudf:.3f}x** | **{all_wins}–{len(measurements) - all_wins}** | **{all_regex_compile * 1000:.3f}** | **{all_cudf_compile * 1000:.4f}** |",
            "",
        ]
    )

    narrowest = min(measurements, key=lambda item: item.speedup)
    largest = max(measurements, key=lambda item: item.speedup)
    summary = (
        f"Regex IR won {all_wins} of {len(measurements)} paired full-corpus cases. "
        f"The narrowest result was {SUITE_LABELS[narrowest.suite]} case "
        f"{narrowest.case} ({narrowest.speedup:.3f}x); the largest was "
        f"{SUITE_LABELS[largest.suite]} case {largest.case} "
        f"({largest.speedup:.3f}x)."
    )
    lines.extend(textwrap.wrap(summary, width=88) + [""])

    for suite in SUITE_ORDER:
        values = sorted(
            (item for item in measurements if item.suite == suite),
            key=lambda item: item.case,
        )
        lines.extend(
            [
                f"#### {SUITE_LABELS[suite]} full-corpus cases",
                "",
                f"[PNG chart](docs/_static/benchmarks/corpus-{suite}-throughput-cases.png) ·",
                f"[SVG chart](docs/_static/benchmarks/corpus-{suite}-throughput-cases.svg) ·",
                f"[case-level CSV](docs/_static/benchmarks/corpus-{suite}-throughput-data.csv)",
                "",
                "| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup | Regex IR JIT-ready (ms) |",
                "|---:|:---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in values:
            lines.append(
                f"| {item.case} | {display_case_name(item.case_name)} | "
                f"{item.input_bytes / (1024**2):.3f} | "
                f"{item.regex_ir_seconds * 1000:.3f} | "
                f"{item.cudf_seconds * 1000:.3f} | {item.speedup:.3f}x | "
                f"{item.regex_ir_jit_ready_seconds * 1000:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "[All complete-corpus measurements](docs/_static/benchmarks/corpus-throughput-data.csv)",
            "",
            README_RESULTS_END,
        ]
    )

    readme = path.read_text(encoding="utf-8")
    begin = readme.index(README_RESULTS_BEGIN)
    end = readme.index(README_RESULTS_END, begin) + len(README_RESULTS_END)
    path.write_text(
        readme[:begin] + "\n".join(lines) + readme[end:], encoding="utf-8"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Render NVIDIA-styled full-corpus Regex IR/cuDF case charts."
    )
    parser.add_argument(
        "nvbench_json",
        type=Path,
        nargs="+",
        help="one or more complete-corpus NVBench JSON exports",
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
        help="README whose generated complete-corpus block should be refreshed",
    )
    return parser.parse_args()


def main() -> None:
    """Generate presentation charts, CSV data, and optional README tables."""

    arguments = parse_arguments()
    measurements = load_measurements(arguments.nvbench_json)
    configure_style()
    for suite in SUITE_ORDER:
        corpus_case_bar_chart(
            measurements, suite, arguments.output_directory
        )
        export_csv(
            [item for item in measurements if item.suite == suite],
            arguments.output_directory / f"corpus-{suite}-throughput-data.csv",
        )
    export_csv(
        measurements, arguments.output_directory / "corpus-throughput-data.csv"
    )
    if arguments.readme is not None:
        write_readme_results(measurements, arguments.readme)


if __name__ == "__main__":
    main()
