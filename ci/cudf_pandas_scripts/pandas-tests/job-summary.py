# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
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
diff_df = pr_df - main_df

pr_df = pr_df[["total", "passed", "failed", "skipped"]]
diff_df = diff_df[["total", "passed", "failed", "skipped"]]
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
    }
)
df = df.sort_values(by=["Failed tests", "Skipped tests"], ascending=False)

print(comment)
print()
print("Here are the results of running the Pandas tests against this PR:")
print()
print(df.to_markdown())
