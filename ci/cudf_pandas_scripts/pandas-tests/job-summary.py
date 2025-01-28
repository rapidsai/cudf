# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pandas as pd


def get_total_and_passed(results):
    total_failed = 0
    total_errored = 0
    total_passed = 0
    for module_name, row in results.items():
        total_failed += row.get("failed", 0)
        total_errored += row.get("errored", 0)
        total_passed += row.get("passed", 0)
    total_tests = total_failed + total_errored + total_passed
    return total_tests, total_passed


main_json = sys.argv[1]
pr_json = sys.argv[2]

# read the results of summarize-test-results.py --summary
with open(main_json) as f:
    main_results = json.load(f)
main_total, main_passed = get_total_and_passed(main_results)

with open(pr_json) as f:
    pr_results = json.load(f)
pr_total, pr_passed = get_total_and_passed(pr_results)

passing_percentage = pr_passed / pr_total * 100
pass_rate_change = abs(pr_passed - main_passed) / main_passed * 100
rate_change_type = "a decrease" if pr_passed < main_passed else "an increase"

comment = (
    "Merging this PR would result in "
    f"{pr_passed}/{pr_total} ({passing_percentage:.2f}%) "
    "Pandas tests passing, "
    f"{rate_change_type} by "
    f"{pass_rate_change:.2f}%. "
    f"Trunk stats: {main_passed}/{main_total}."
)


def emoji_passed(x):
    if x > 0:
        return f"{x}✅"
    elif x < 0:
        return f"{x}❌"
    else:
        return f"{x}"


def emoji_failed(x):
    if x > 0:
        return f"{x}❌"
    elif x < 0:
        return f"{x}✅"
    else:
        return f"{x}"


# convert pr_results to a pandas DataFrame and then a markdown table
pr_df = pd.DataFrame.from_dict(pr_results, orient="index").sort_index()
main_df = pd.DataFrame.from_dict(main_results, orient="index").sort_index()
total_usage = main_df["_slow_function_call"] + main_df["_fast_function_call"]
main_df["CPU Usage"] = (
    (main_df["_slow_function_call"] / total_usage) * 100.0
).round(1)
main_df["GPU Usage"] = (
    (main_df["_fast_function_call"] / total_usage) * 100.0
).round(1)

total_usage = pr_df["_slow_function_call"] + pr_df["_fast_function_call"]
pr_df["CPU Usage"] = (
    (pr_df["_slow_function_call"] / total_usage) * 100.0
).round(1)
pr_df["GPU Usage"] = (
    (pr_df["_fast_function_call"] / total_usage) * 100.0
).round(1)

cpu_usage_mean = pr_df["CPU Usage"].mean().round(2)
gpu_usage_mean = pr_df["GPU Usage"].mean().round(2)

gpu_usage_rate_change = abs(
    pr_df["GPU Usage"].mean() - main_df["GPU Usage"].mean()
)
pr_df["CPU Usage"] = pr_df["CPU Usage"].fillna(0)
pr_df["GPU Usage"] = pr_df["GPU Usage"].fillna(0)
main_df["CPU Usage"] = main_df["CPU Usage"].fillna(0)
main_df["GPU Usage"] = main_df["GPU Usage"].fillna(0)

diff_df = pr_df - main_df
diff_df["CPU Usage"] = diff_df["CPU Usage"].round(1).fillna(0)
diff_df["GPU Usage"] = diff_df["GPU Usage"].round(1).fillna(0)

# Add '%' suffix to "CPU Usage" and "GPU Usage" columns
pr_df["CPU Usage"] = pr_df["CPU Usage"].astype(str) + "%"
pr_df["GPU Usage"] = pr_df["GPU Usage"].astype(str) + "%"

pr_df = pr_df[
    ["total", "passed", "failed", "skipped", "CPU Usage", "GPU Usage"]
]
diff_df = diff_df[
    ["total", "passed", "failed", "skipped", "CPU Usage", "GPU Usage"]
]
diff_df.columns = diff_df.columns + "_diff"
diff_df["passed_diff"] = diff_df["passed_diff"].map(emoji_passed)
diff_df["failed_diff"] = diff_df["failed_diff"].map(emoji_failed)
diff_df["skipped_diff"] = diff_df["skipped_diff"].map(emoji_failed)

df = pd.concat([pr_df, diff_df], axis=1)
df = df.rename_axis("Test module")

df = df.rename(
    columns={
        "total": "Total tests",
        "passed": "Passed tests",
        "failed": "Failed tests",
        "skipped": "Skipped tests",
        "total_diff": "Total delta",
        "passed_diff": "Passed delta",
        "failed_diff": "Failed delta",
        "skipped_diff": "Skipped delta",
        "CPU Usage_diff": "CPU Usage delta",
        "GPU Usage_diff": "GPU Usage delta",
    }
)
df = df.sort_values(by=["CPU Usage delta", "Total tests"], ascending=False)
df["CPU Usage delta"] = df["CPU Usage delta"].map(emoji_failed)
df["GPU Usage delta"] = df["GPU Usage delta"].map(emoji_passed)
df = df[
    [
        "Total tests",
        "CPU Usage delta",
        "GPU Usage delta",
        "Passed tests",
        "Failed tests",
        "Skipped tests",
        "CPU Usage",
        "GPU Usage",
        "Total delta",
        "Passed delta",
        "Failed delta",
        "Skipped delta",
    ]
]
print(comment)
print()
print(
    f"Average GPU usage: {gpu_usage_mean}% {'an increase' if gpu_usage_rate_change > 0 else 'a decrease'} by {gpu_usage_rate_change}%"
)
print()
print(f"Average CPU usage: {cpu_usage_mean}%")
print()
print("Here are the results of running the Pandas tests against this PR:")
print()
print(df.to_markdown())
