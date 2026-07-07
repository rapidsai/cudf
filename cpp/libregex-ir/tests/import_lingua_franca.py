#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream string patterns from Lingua Franca production-regex NDJSON.

Usage:
  tests/import_lingua_franca.py uniq-regexes-8.json patterns.txt

The external corpus is intentionally not downloaded or vendored by this project.
"""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    with (
        open(args.input, "r", encoding="utf-8") as source,
        open(args.output, "w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
                pattern = record.get("pattern")
                if isinstance(pattern, dict):
                    pattern = pattern.get("pattern")
                if not isinstance(pattern, str):
                    continue
                destination.write(
                    "-"
                    + pattern.encode("utf-8", errors="surrogatepass").hex()
                    + "\n"
                )
            except (json.JSONDecodeError, UnicodeError) as error:
                raise SystemExit(f"line {line_number}: {error}") from error


if __name__ == "__main__":
    main()
