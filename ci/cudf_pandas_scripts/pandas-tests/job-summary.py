# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pandas as pd


def get_total_and_passed(results):
    total_failed = 0
    total_errored = 0
    total_passed = 0
    total_skipped = 0
    total_xfailed_by_cudf_pandas = 0
    total_skipped_by_cudf_pandas = 0
    for module_name, row in results.items():
        total_failed += row.get("failed", 0)
        total_errored += row.get("errored", 0)
        total_passed += row.get("passed", 0)
        total_skipped += row.get("skipped", 0)
        total_xfailed_by_cudf_pandas += row.get("xfailed_by_cudf_pandas", 0)
        total_skipped_by_cudf_pandas += row.get("skipped_by_cudf_pandas", 0)
    total_tests = total_failed + total_errored + total_passed + total_skipped
    return (
        total_tests,
        total_passed,
        total_xfailed_by_cudf_pandas,
        total_skipped_by_cudf_pandas,
        total_skipped,
    )


main_json = sys.argv[1]
pr_json = sys.argv[2]
branch_version = sys.argv[3]

# read the results of summarize-test-results.py --summary
with open(main_json) as f:
    main_results = json.load(f)
(
    main_total,
    main_passed,
    main_xfailed_by_cudf_pandas,
    main_skipped_by_cudf_pandas,
    main_skipped,
) = get_total_and_passed(main_results)

with open(pr_json) as f:
    pr_results = json.load(f)
(
    pr_total,
    pr_passed,
    pr_xfailed_by_cudf_pandas,
    pr_skipped_by_cudf_pandas,
    pr_skipped,
) = get_total_and_passed(pr_results)

passing_percentage = pr_passed / pr_total * 100


metrics_df = pd.DataFrame(
    {
        "This PR": [
            pr_total,
            pr_passed,
            pr_skipped_by_cudf_pandas,
            pr_xfailed_by_cudf_pandas,
            pr_skipped
            - (pr_skipped_by_cudf_pandas + pr_xfailed_by_cudf_pandas),
        ],
        f"branch-{branch_version}": [
            main_total,
            main_passed,
            main_skipped_by_cudf_pandas,
            main_xfailed_by_cudf_pandas,
            main_skipped
            - (main_skipped_by_cudf_pandas + main_xfailed_by_cudf_pandas),
        ],
    },
    index=[
        "Total tests",
        "Passed tests",
        "cudf.Pandas Skipped",
        "cudf.Pandas xFailed",
        "pandas skipped",
    ],
)


def emoji_passed(x):
    """Format number with emoji: positive -> ✅, negative -> ❌"""
    if x > 0:
        return f"{x}✅"
    elif x < 0:
        return f"{x}❌"
    else:
        return f"{x}"


def emoji_failed(x):
    """Format number with emoji: positive -> ❌, negative -> ✅ (inverse of emoji_passed)"""
    if x > 0:
        return f"{x}❌"
    elif x < 0:
        return f"{x}✅"
    else:
        return f"{x}"


# convert pr_results to a pandas DataFrame and then a markdown table
pr_df = pd.DataFrame.from_dict(pr_results, orient="index").sort_index()
main_df = pd.DataFrame.from_dict(main_results, orient="index").sort_index()
# Calculate CPU and GPU usage percentages for main branch
total_usage = main_df["_slow_function_call"] + main_df["_fast_function_call"]
main_df["CPU Usage"] = (
    (main_df["_slow_function_call"] / total_usage) * 100.0
).round(1)
main_df["GPU Usage"] = (
    (main_df["_fast_function_call"] / total_usage) * 100.0
).round(1)

# Calculate CPU and GPU usage percentages for PR
total_usage = pr_df["_slow_function_call"] + pr_df["_fast_function_call"]
pr_df["CPU Usage"] = (
    (pr_df["_slow_function_call"] / total_usage) * 100.0
).round(1)
pr_df["GPU Usage"] = (
    (pr_df["_fast_function_call"] / total_usage) * 100.0
).round(1)

# Calculate average usages
cpu_usage_mean = pr_df["CPU Usage"].mean().round(2)
gpu_usage_mean = pr_df["GPU Usage"].mean().round(2)
gpu_usage_rate_change = (
    pr_df["GPU Usage"].mean() - main_df["GPU Usage"].mean()
).round(2)

# Handle NaN values
pr_df["CPU Usage"] = pr_df["CPU Usage"].fillna(0)
pr_df["GPU Usage"] = pr_df["GPU Usage"].fillna(0)
main_df["CPU Usage"] = main_df["CPU Usage"].fillna(0)
main_df["GPU Usage"] = main_df["GPU Usage"].fillna(0)

# Calculate differences between PR and main
diff_df = pr_df - main_df
diff_df["CPU Usage"] = diff_df["CPU Usage"].round(1).fillna(0)
diff_df["GPU Usage"] = diff_df["GPU Usage"].round(1).fillna(0)

# Add '%' suffix to "CPU Usage" and "GPU Usage" columns
pr_df["CPU Usage"] = pr_df["CPU Usage"].astype(str) + "%"
pr_df["GPU Usage"] = pr_df["GPU Usage"].astype(str) + "%"

# Select relevant columns
pr_df = pr_df[["total", "CPU Usage", "GPU Usage"]]
diff_df = diff_df[["total", "CPU Usage", "GPU Usage"]]

# Rename diff columns to indicate they are differences
diff_df.columns = diff_df.columns + "_diff"

# Combine PR results with differences
df = pd.concat([pr_df, diff_df], axis=1)
df = df.rename_axis("Test module")

# Rename columns for better readability
df = df.rename(
    columns={
        "total": "Total tests",
        "total_diff": "Total delta",
        "CPU Usage_diff": "CPU Usage delta",
        "GPU Usage_diff": "GPU Usage delta",
    }
)

# Sort by CPU usage delta and total tests
df = df.sort_values(by=["CPU Usage delta", "Total tests"], ascending=False)

# Apply emoji formatting to usage deltas
df["CPU Usage delta"] = df["CPU Usage delta"].map(emoji_failed)
df["GPU Usage delta"] = df["GPU Usage delta"].map(emoji_passed)

# Select final columns to display
df = df[
    [
        "Total tests",
        "CPU Usage delta",
        "GPU Usage delta",
        "CPU Usage",
        "GPU Usage",
        "Total delta",
    ]
]
# Print summary and results
print(metrics_df.to_markdown())
print()
print(
    f"Average GPU usage: {gpu_usage_mean}% ({gpu_usage_rate_change:+.2f}% change from trunk)"
)
print(f"Average CPU usage: {cpu_usage_mean}%")
print()
print("Here are the results of running the Pandas tests against this PR:")
print()
print(df.to_markdown())
