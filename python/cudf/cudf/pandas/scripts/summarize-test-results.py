# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Summarizes the test results per module.

Examples
--------
    python summarize-test-results.py log.json
    python summarize-test-results.py log.json --output json
    python summarize-test-results.py log.json --output table
"""

import argparse
import glob
import json
import os

from rich.console import Console
from rich.table import Table

PANDAS_TEST_PREFIX = "pandas-tests/"


def get_per_module_results(log_file_name):
    per_module_results = {}
    with open(log_file_name) as f:
        for line in f:
            try:
                line = json.loads(line)
            except Exception:
                line = {}
            if "outcome" in line:
                outcome = line["outcome"]
                # outcome can be "passed", "failed", or "skipped".
                # Depending on other fields, it can indicate
                # an errored, xpassed, or xfailed test.
                if line.get("when", None) != "call":
                    # when != call indicates test setup or teardown
                    if outcome == "failed":
                        # if the test failed during setup or teardown,
                        # it counts as an "errored" test:
                        outcome = "errored"
                    else:
                        # we don't care about other outcomes during
                        # setup or teardown
                        continue
                else:
                    if line.get("wasxfail", False) and outcome == "passed":
                        # it's an xpassed test
                        outcome = "failed"
                module_name = (
                    line["nodeid"]
                    .split("::")[0]
                    .removeprefix(PANDAS_TEST_PREFIX)
                )
                per_module_results.setdefault(module_name, {})
                per_module_results[module_name].setdefault("total", 0)
                per_module_results[module_name].setdefault(outcome, 0)
                per_module_results[module_name]["total"] += 1
                per_module_results[module_name][outcome] += 1

    directory = os.path.dirname(log_file_name)
    pattern = os.path.join(directory, "function_call_counts_worker_*.json")
    matching_files = glob.glob(pattern)
    function_call_counts = {}

    for file in matching_files:
        with open(file) as f:
            function_call_count = json.load(f)
        if not function_call_counts:
            function_call_counts.update(function_call_count)
        else:
            for key, value in function_call_count.items():
                if key not in function_call_counts:
                    function_call_counts[key] = value
                else:
                    if "_slow_function_call" not in function_call_counts[key]:
                        function_call_counts[key]["_slow_function_call"] = 0
                    if "_fast_function_call" not in function_call_counts[key]:
                        function_call_counts[key]["_fast_function_call"] = 0
                    function_call_counts[key]["_slow_function_call"] += (
                        value.get("_slow_function_call", 0)
                    )
                    function_call_counts[key]["_fast_function_call"] += (
                        value.get("_fast_function_call", 0)
                    )

    for key, value in per_module_results.items():
        if key in function_call_counts:
            per_module_results[key]["_slow_function_call"] = (
                function_call_counts[key].get("_slow_function_call", 0)
            )
            per_module_results[key]["_fast_function_call"] = (
                function_call_counts[key].get("_fast_function_call", 0)
            )
        else:
            per_module_results[key]["_slow_function_call"] = 0
            per_module_results[key]["_fast_function_call"] = 0
    return per_module_results


def sort_results(results):
    sorted_keys = sorted(
        results, key=lambda key: results[key].get("failed", 0)
    )
    return {key: results[key] for key in sorted_keys}


def print_results_as_json(results):
    print(json.dumps(results, indent=4))


def print_results_as_table(results):
    table = Table()
    table.add_column("Test module")
    table.add_column("Total tests")
    table.add_column("Passed tests")
    table.add_column("Failed tests")
    table.add_column("Errored tests")
    table.add_column("Skipped tests")
    totals = {"total": 0, "passed": 0, "failed": 0, "errored": 0, "skipped": 0}
    for module_name, row in results.items():
        values = []
        for key in ("total", "passed", "failed", "errored", "skipped"):
            totals[key] += row.get(key, 0)
            values.append(row.get(key, 0))
        table.add_row(module_name, *map(str, values))
    table.add_section()
    table.add_row(
        "total={}, passed={}, failed={}, errored={}, skipped={}".format(
            *map(str, totals.values())
        )
    )
    console = Console()
    console.print(table)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_file_name", nargs=1, help="The input log file name"
    )
    parser.add_argument(
        "--output",
        choices=["json", "table"],
        default="table",
        help="The output format",
    )
    args = parser.parse_args()
    results = sort_results(get_per_module_results(args.log_file_name[0]))
    if args.output == "json":
        print_results_as_json(results)
    else:
        print_results_as_table(results)
