#!/usr/bin/env python3
"""
cudfgrep: GPU-accelerated grep utility using cuDF/nvtext

- Grep plain text files (one string per line, string only)
- Multiple matches per line are supported
- CLI matches grep: -e, -i, -c, etc.
- Future: column mode for CSV/Parquet, GDS enablement

Benchmarking:
- Run with and without --gds (or GDS=1 env var)
- Test on PCIe and Grace systems for throughput

Example usage:
  cudfgrep -e PATTERN file.txt
  cudfgrep -e PATTERN --gds file.txt
"""
import argparse
import os
import sys
import cudf
import cupy as cp


def parse_args():
    parser = argparse.ArgumentParser(
        prog="cudfgrep",
        description="GPU-accelerated grep utility using cuDF/nvtext. Grep plain text files (one string per line). Multiple matches per line supported.",
        usage="%(prog)s [OPTIONS] -e PATTERN [FILE ...]"
    )
    parser.add_argument("-e", "--regexp", dest="pattern", required=True, help="Pattern to search for (required)")
    parser.add_argument("-i", "--ignore-case", dest="ignore_case", action="store_true", help="Ignore case distinctions")
    parser.add_argument("-c", "--count", dest="count", action="store_true", help="Only print a count of matching lines per file")
    parser.add_argument("--gds", dest="gds", action="store_true", help="Enable GDS (GPUDirect Storage)")
    parser.add_argument("files", nargs="+", help="Input text files (one string per line)")
    return parser.parse_args()


def main():
    args = parse_args()
    # Enable GDS if requested
    if args.gds:
        os.environ["CUDF_GDS"] = "1"
    # Read all lines from all files
    lines = []
    for fname in args.files:
        with open(fname, "r", encoding="utf-8") as f:
            lines.extend(line.rstrip("\n") for line in f)
    if not lines:
        return
    s = cudf.Series(lines)
    # Compile regex
    flags = 0
    if args.ignore_case:
        flags |= cp.str_._regex_compile_option_ignorecase
    # Find all matches per line
    matches = s.str.findall(args.pattern, flags=flags)
    if args.count:
        print(sum(bool(m) for m in matches))
        return
    for i, m in enumerate(matches):
        if m:
            for match in m:
                print(f"{lines[i]}\t{match}")

if __name__ == "__main__":
    main()
