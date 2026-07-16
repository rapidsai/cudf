# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""GPU-accelerated grep built on cuDF string operations."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cudf

__all__ = ["grep", "main"]

_STDIN_NAME = "(standard input)"

# (1-based line number, text) for a single result row.
Match = tuple[int, str]


def _emit(text: str) -> None:
    """Write a single line to stdout."""
    sys.stdout.write(text + "\n")


def _combine_patterns(
    patterns: Sequence[str], word: bool, whole_line: bool
) -> str:
    """Combine one or more raw patterns into a single regex string.

    Multiple patterns (repeated ``-e``) are OR-ed together, matching grep.
    ``-w`` / ``-x`` add word / whole-line anchors.
    """
    if not patterns:
        raise ValueError("no pattern given")
    combined = "|".join(f"(?:{pat})" for pat in patterns)
    if word:
        combined = rf"\b(?:{combined})\b"
    if whole_line:
        combined = rf"^(?:{combined})$"
    return combined


def _configure_gds(gds: bool | None) -> None:
    """Toggle GPUDirect Storage via KvikIO's compatibility mode.

    ``gds`` is ``True`` (``--gds``), ``False`` (``--no-gds``) or ``None``
    (unset, honour the ``CUDFGREP_GDS`` env var). Enabling GDS forces the
    KvikIO cuFile path (``KVIKIO_COMPAT_MODE=OFF``); disabling it selects the
    POSIX path (``KVIKIO_COMPAT_MODE=ON``).

    This must run before cuDF (and therefore KvikIO) is imported, which is why
    cuDF is imported lazily throughout this module.
    """
    if gds is None:
        env = os.environ.get("CUDFGREP_GDS")
        if env is None:
            return
        gds = env.strip().lower() in ("1", "true", "on", "yes")
    os.environ["KVIKIO_COMPAT_MODE"] = "OFF" if gds else "ON"


def _read_lines(path: str) -> cudf.Series:
    """Load a text file as one string per row on the GPU."""
    import cudf

    series = cudf.read_text(path, delimiter="\n", strip_delimiters=True)
    # A trailing newline yields an empty final row; drop it to match grep.
    if len(series) and series.iloc[-1] == "":
        series = series.iloc[:-1]
    return series.reset_index(drop=True)


def _lines_from_stdin() -> cudf.Series:
    """Load stdin as one string per row on the GPU."""
    import cudf

    data = sys.stdin.read().split("\n")
    if data and data[-1] == "":
        data.pop()
    return cudf.Series(data, dtype="str")


def _search(
    series: cudf.Series,
    pattern: str,
    flags: int,
    only_matching: bool,
    invert: bool,
) -> list[Match]:
    """Return matches as ``(line_number, text)`` pairs.

    With ``only_matching`` each matched substring is returned separately
    (supporting multiple matches per line); otherwise whole matching lines
    are returned.
    """
    if only_matching:
        # grep prints nothing for ``-o -v``: there are no matching lines to
        # extract matched parts from.
        if invert:
            return []
        found = series.str.findall(pattern, flags=flags).explode()
        found = found[found.notna()]
        host = found.to_pandas()
        return [(int(idx) + 1, str(val)) for idx, val in host.items()]

    mask = series.str.contains(pattern, flags=flags, regex=True).fillna(False)
    if invert:
        mask = ~mask
    host = series[mask].to_pandas()
    return [(int(idx) + 1, str(val)) for idx, val in host.items()]


def _count(series: cudf.Series, pattern: str, flags: int, invert: bool) -> int:
    """Count matching lines (``grep -c`` semantics)."""
    mask = series.str.contains(pattern, flags=flags, regex=True).fillna(False)
    if invert:
        mask = ~mask
    return int(mask.sum())


def grep(
    pattern: str,
    filepath: str,
    *,
    ignore_case: bool = False,
    invert: bool = False,
    only_matching: bool = False,
    word: bool = False,
    whole_line: bool = False,
) -> list[Match]:
    """Search ``filepath`` for ``pattern`` on the GPU.

    Parameters
    ----------
    pattern : str
        Regular expression to search for.
    filepath : str
        Path to a text file, treated as one string per line.
    ignore_case : bool, default False
        Match case-insensitively (``grep -i``).
    invert : bool, default False
        Select non-matching lines (``grep -v``).
    only_matching : bool, default False
        Return each matched substring separately, supporting multiple
        matches per line (``grep -o``).
    word : bool, default False
        Match whole words only (``grep -w``).
    whole_line : bool, default False
        Match whole lines only (``grep -x``).

    Returns
    -------
    list of (int, str)
        ``(line_number, text)`` pairs, where ``text`` is the matching line,
        or the matched substring when ``only_matching`` is set.
    """
    combined = _combine_patterns([pattern], word, whole_line)
    flags = re.IGNORECASE if ignore_case else 0
    series = _read_lines(filepath)
    return _search(series, combined, flags, only_matching, invert)


def _run_benchmark(
    path: str, pattern: str, flags: int, repeats: int = 3
) -> None:
    """Time load+search and report throughput in GB/s."""
    size = os.path.getsize(path)

    # Warm-up run (allocator, JIT, file cache) is not timed.
    matches = _count(_read_lines(path), pattern, flags, invert=False)

    best = float("inf")
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        matches = _count(_read_lines(path), pattern, flags, invert=False)
        best = min(best, time.perf_counter() - start)

    gbps = (size / best / 1e9) if best > 0 else float("inf")
    _emit(
        f"backend=gpu  size={size / 1e6:.1f} MB  "
        f"time={best * 1e3:.1f} ms  throughput={gbps:.3f} GB/s  "
        f"matches={matches}"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the grep-compatible argument parser."""
    # add_help=False so -h maps to --no-filename, matching grep.
    parser = argparse.ArgumentParser(
        prog="cudfgrep",
        add_help=False,
        description="GPU-accelerated grep powered by cuDF.",
        usage="cudfgrep [OPTIONS] PATTERN [FILE...]",
    )
    parser.add_argument(
        "--help", action="help", help="show this help message and exit"
    )
    parser.add_argument("pattern", nargs="?", help="regex pattern")
    parser.add_argument(
        "files", nargs="*", help="files to search ('-' for stdin)"
    )
    parser.add_argument(
        "-e",
        "--regexp",
        action="append",
        metavar="PATTERN",
        dest="regexp",
        help="pattern to search for (may be repeated)",
    )
    parser.add_argument(
        "-i", "--ignore-case", action="store_true", help="ignore case"
    )
    parser.add_argument(
        "-n",
        "--line-number",
        action="store_true",
        help="prefix each match with its line number",
    )
    parser.add_argument(
        "-c",
        "--count",
        action="store_true",
        help="print only a count of matching lines",
    )
    parser.add_argument(
        "-v",
        "--invert-match",
        dest="invert",
        action="store_true",
        help="select non-matching lines",
    )
    parser.add_argument(
        "-o",
        "--only-matching",
        action="store_true",
        help="print only the matched parts of a line",
    )
    parser.add_argument(
        "-w",
        "--word-regexp",
        dest="word",
        action="store_true",
        help="match whole words only",
    )
    parser.add_argument(
        "-x",
        "--line-regexp",
        dest="whole_line",
        action="store_true",
        help="match whole lines only",
    )
    parser.add_argument(
        "-H",
        "--with-filename",
        action="store_true",
        help="print the file name for each match",
    )
    parser.add_argument(
        "-h",
        "--no-filename",
        action="store_true",
        help="suppress the file name prefix",
    )
    parser.add_argument(
        "--gds",
        dest="gds",
        action="store_true",
        default=None,
        help="enable GPUDirect Storage",
    )
    parser.add_argument(
        "--no-gds",
        dest="gds",
        action="store_false",
        help="disable GPUDirect Storage",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="report throughput (GB/s) instead of matches",
    )
    return parser


def _gather_patterns(args: argparse.Namespace) -> list[str]:
    """Collect patterns from ``-e`` options and/or the positional argument."""
    patterns: list[str] = []
    if args.regexp:
        patterns.extend(args.regexp)
        # With -e, the positional 'pattern' is actually a file.
        if args.pattern is not None:
            args.files = [args.pattern, *args.files]
            args.pattern = None
    elif args.pattern is not None:
        patterns.append(args.pattern)
    return patterns


def _display_name(path: str) -> str:
    """Return the name to show for a path ('-' means stdin)."""
    return _STDIN_NAME if path == "-" else path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cudfgrep command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    raw_patterns = _gather_patterns(args)
    if not raw_patterns:
        parser.error("no pattern given")

    try:
        pattern = _combine_patterns(raw_patterns, args.word, args.whole_line)
        re.compile(pattern)
    except (re.error, ValueError) as exc:
        sys.stderr.write(f"cudfgrep: invalid pattern: {exc}\n")
        return 2

    flags = re.IGNORECASE if args.ignore_case else 0
    # Configure GDS before any cuDF import so the setting takes effect.
    _configure_gds(args.gds)

    files = args.files or ["-"]
    show_name = args.with_filename or (len(files) > 1 and not args.no_filename)

    if args.benchmark:
        rc = 0
        for path in files:
            if path == "-":
                sys.stderr.write(
                    "cudfgrep: --benchmark needs a file, not stdin\n"
                )
                rc = 2
                continue
            try:
                _run_benchmark(path, pattern, flags)
            except OSError as exc:
                sys.stderr.write(f"cudfgrep: {path}: {exc}\n")
                rc = 2
        return rc

    any_match = False
    had_error = False
    for path in files:
        try:
            series = _lines_from_stdin() if path == "-" else _read_lines(path)
        except OSError as exc:
            sys.stderr.write(f"cudfgrep: {path}: {exc}\n")
            had_error = True
            continue

        name = _display_name(path)

        if args.count:
            n = _count(series, pattern, flags, args.invert)
            _emit(f"{name}:{n}" if show_name else str(n))
            any_match = any_match or n > 0
            continue

        results = _search(
            series, pattern, flags, args.only_matching, args.invert
        )
        if results:
            any_match = True
        for lineno, text in results:
            parts: list[str] = []
            if show_name:
                parts.append(name)
            if args.line_number:
                parts.append(str(lineno))
            parts.append(text)
            _emit(":".join(parts))

    if had_error:
        return 2
    return 0 if any_match else 1
