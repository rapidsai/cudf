---
name: debug-cudf-pandas
description: Debug and fix pandas test suite failures under the cudf.pandas compatibility layer. Use when given pytest node IDs of failing pandas tests that need to be fixed for cudf.pandas compatibility.
---

# Debug cudf.pandas Compatibility Failures

## Overview

`cudf.pandas` is a zero-code-change GPU accelerator for pandas. It intercepts `import pandas` via a Python `MetaPathFinder` (`python/cudf/cudf/pandas/module_accelerator.py`) and wraps all pandas objects and functions in proxy types that try the cudf (GPU) path first, falling back to pandas (CPU) if cudf raises an exception.

When the pandas test suite is run with `-p cudf.pandas`, test failures indicate one of:
- A **cudf implementation bug** — cudf's DataFrame/Series/Column/Index behavior differs from pandas for an edge case
- A **proxy/dispatch bug** — the wrapping/unwrapping mechanism doesn't correctly handle a type or operation
- A **missing proxy registration** — a pandas type or return value has no registered cudf equivalent
- A **to/from_pandas conversion bug** — data is corrupted or lost when converting between cudf and pandas objects
- A **test setup bug** — the testing scripts or conftest-patch introduce an issue
- A **pandas bug** — rarely, the expected behavior in the pandas test itself is wrong

Your job is to find the root cause and implement the fix.

---

## Prerequisites

Before starting, verify you are at the repository root. All commands in this skill assume the working directory is the cudf repository root.

**Clean up any previous test run state.** The test runner appends `conftest-patch.py` to the pandas conftest on every invocation. If `pandas-testing/pandas-tests/` already exists from a prior run, the conftest will have duplicate hook registrations and cause spurious errors. Always delete it before running:

```bash
rm -rf pandas-testing/pandas-tests/
```

The cudf Python package is almost entirely pure Python. For **inplace installs** (e.g. `pip install -e .`), changes to `.py` files take effect immediately — no rebuild is needed. For non-inplace installs (e.g. `./build.sh`), you must either reinstall or copy changed files to site-packages.

---

## Input Format

Input is one or more pytest node IDs, relative to the `pandas-tests/` directory inside the test harness:

```
tests/groupby/test_reductions.py::test_first_last_skipna[Float64-False-False-first]
tests/frame/methods/test_fillna.py::test_fillna_inplace
```

Node IDs with parameters like `[Float64-False-False-first]` target a specific parametrized case. When multiple node IDs are provided, check whether they share a root cause before attempting separate fixes.

---

## Step 0 — Update conftest-patch.py

The file `python/cudf/cudf/pandas/scripts/conftest-patch.py` contains three dictionaries that gate how tests are handled:

- **`NODEIDS_THAT_FAIL`** — tests marked `xfail` (expected to fail). Keys are alphabetically sorted.
- **`NODEIDS_TO_SKIP`** — tests marked `skip` (not run at all). Keys are alphabetically sorted.
- **`NODEIDS_PATHS_TO_SKIP`** — prefix-based path skips covering entire modules.

Because the test runner sets `xfail_strict = true`, a test listed in `NODEIDS_THAT_FAIL` that unexpectedly *passes* is reported as `XPASS` — which is also a failure. You must remove the entry before testing your fix, or you will never see a genuine pass.

Search for the node ID:

```bash
grep -n "tests/groupby/test_reductions.py::test_first_last_skipna" \
    python/cudf/cudf/pandas/scripts/conftest-patch.py
```

If found, remove the line. Keys must remain in alphabetical order after the edit. Then validate the file still parses:

```bash
python -c "exec(open('python/cudf/cudf/pandas/scripts/conftest-patch.py').read())"
```

If the node ID is *not* found in any dictionary, you are dealing with a new regression — proceed directly to Step 1.

---

## Step 1 — Reproduce the Failure

Run from the repo root:

```bash
rm -rf pandas-testing/pandas-tests/
bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "tests/groupby/test_reductions.py::test_first_last_skipna[Float64-False-False-first]" \
    -xvs
```

The script clones the matching pandas version, copies tests, appends the conftest patch, and runs pytest with `-p cudf.pandas`. Substitute your actual node ID.

**If the test passes**: the xfail entry was stale. Commit only the `conftest-patch.py` change and stop.

**If the test fails**: read the failure output carefully — the assertion message tells you the exact behavioral difference. Proceed to Step 2.

---

## Step 2 — Understand the Test

After the first run, the test file exists at:

```
pandas-testing/pandas-tests/tests/<module>/<test_file>.py
```

Read the test. Identify:
- Which pandas API is being exercised (method name, class, module)
- What the test asserts — this is the exact behavior cudf must match
- Whether parameters narrow the scope (e.g. only `Float64` dtype, only `skipna=False`)

The assertion error output from Step 1 tells you what cudf produced vs. what was expected. Use both together to understand the gap.

---

## Step 3 — Diagnose Root Cause

**Always check cudf's direct behavior first.** Do not jump to proxy investigation.

### 3a. Direct cudf test (most important step)

Write a minimal script that exercises the API using cudf directly, without going through `cudf.pandas`. This is strictly an example — adapt the structure to match what the failing test actually does (comparisons, exception checks, dtype validation, etc.):

```python
import cudf
import pandas as pd

# Mirror the test setup
pdf = pd.DataFrame({"value": [1.0, None, 2.0, 3.0]}, dtype="Float64")
cdf = cudf.from_pandas(pdf)

pd_result = pdf.groupby(level=0).first(skipna=False)
cudf_result = cdf.groupby(level=0).first(skipna=False)

print("pandas:", pd_result)
print("cudf:  ", cudf_result)
print("match:", pd_result.equals(cudf_result.to_pandas()))
```

For tests that assert exceptions are raised, structure your script to verify both cudf and pandas raise the same exception type and message.

Save to a temporary file (e.g. `test_debug.py`) and run:

```bash
python test_debug.py
```

Results:
- **cudf result differs from pandas** → cudf implementation bug → go to Step 4a
- **cudf raises an exception** → missing feature or bug → evaluate scope; may need user input if the feature is large. Note: this may be OK if the test is verifying that an exception *should* be raised.
- **cudf result matches pandas** → proxy/dispatch bug → go to Step 4b

### 3b. Environment variable diagnostics (run through cudf.pandas)

Use these env vars to trace what is happening at the proxy layer:

| Variable | Effect |
|----------|--------|
| `CUDF_PANDAS_FAIL_ON_FALLBACK=1` | Raises instead of silently falling back — shows exactly which operation triggers fallback |
| `LOG_FAST_FALLBACK=1` | Logs every fallback with function name and exception |
| `CUDF_PANDAS_DEBUGGING=1` | Runs both cudf and pandas paths in parallel, warns on result divergence |

```bash
rm -rf pandas-testing/pandas-tests/
CUDF_PANDAS_FAIL_ON_FALLBACK=1 bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "<node_id>" -xvs
```

```bash
rm -rf pandas-testing/pandas-tests/
LOG_FAST_FALLBACK=1 bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "<node_id>" -xvs 2>&1 | grep -i fallback
```

### 3c. Standalone instrumented script

For deeper investigation, copy the relevant test body into a temporary script (e.g. `test_debug.py`), add print statements or assertions at intermediate steps, then run through the proxy layer or try to write matching pandas code to see the differences in behaviors and fix them:

```bash
python -m cudf.pandas test_debug.py
```

This gives you full control to narrow down exactly where the divergence begins.

---

## Step 4a — Fix a cudf Implementation Bug

The fix must be **tightly scoped** and try to keep it minimal. Fixing one edge case can break another test. Do not refactor surrounding code while fixing, and do not add `mode.pandas_compatible` guards (ask the user first).

**Classify the failure mode** before writing a fix: is it a specific issue (e.g. one method handles a keyword incorrectly) or a broad failure mode (e.g. dtype casting inconsistency that affects many operations)? For broad issues, consider whether the fix should be applied at a shared/base level rather than patching individual methods.

**First check if the problem exists in cudf classic** (i.e. `import cudf` without `cudf.pandas`). If the bug is in cudf core, fix it there. Only move to cudf.pandas-specific fixes if cudf classic behaves correctly.

Common fix locations:

| What fails | Where to look |
|---|---|
| DataFrame method | `python/cudf/cudf/core/dataframe.py` |
| Series method | `python/cudf/cudf/core/series.py` |
| Index operation | `python/cudf/cudf/core/index.py` |
| Shared DataFrame/Series method | `python/cudf/cudf/core/indexed_frame.py` |
| Column-level operation | `python/cudf/cudf/core/column/*.py` |
| GroupBy operation | `python/cudf/cudf/core/groupby/` |
| IO operation | `python/cudf/cudf/io/` |

When a fix touches column operations that call into `pylibcudf`, consult `python/CLAUDE.md` for the required buffer access pattern (`column.access(mode="read", scope="internal")`).

Note: `mode.pandas_compatible` is automatically set to `True` when cudf.pandas is active. Account for this in any conditional logic, but do not add new guards for it without explicit user approval.

---

## Step 4b — Fix a Proxy/Dispatch Bug

Only reach this step after Step 3a has confirmed that cudf itself is correct.

**Most common cause**: a pandas return type has no registered cudf proxy. Check `python/cudf/cudf/pandas/_wrappers/pandas.py` — this file registers which pandas types map to which cudf types using `make_final_proxy_type()` and related functions. If a new pandas type needs wrapping, add the registration here.

**`fast_slow_proxy.py` and `module_accelerator.py`** are core infrastructure files. Fix them only if you believe the bug is in one of them.

---

## Step 5 — Verify the Fix

Three checks are required. Run them in order.

**a. Target test passes:**

```bash
rm -rf pandas-testing/pandas-tests/
bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh "<node_id>" -xvs
```

Expected: exit code 0, test shows `PASSED`.

**b. Fix runs on GPU (no silent fallback):**

```bash
rm -rf pandas-testing/pandas-tests/
CUDF_PANDAS_FAIL_ON_FALLBACK=1 bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "<node_id>" -xvs
```

Expected: test still passes. If this fails, the fix works by falling back to pandas rather than actually fixing cudf — that is not acceptable.

Exception: some tests intentionally validate fallback behavior. If `FAIL_ON_FALLBACK` causes this test to fail but the test logic requires fallback, skip this check for that specific test and note the justification.

**c. No regressions in the module:**

```bash
rm -rf pandas-testing/pandas-tests/
bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "tests/<module_directory>/" --tb=line -q
```

Replace `<module_directory>` with the directory containing your test (e.g. `tests/groupby/`). Any new failures that are not already listed in `conftest-patch.py` must be investigated before committing.

**d. Add unit tests (where appropriate):**

If your fix changes cudf behavior, add a unit test in `cudf` classic or `cudf.pandas` depending on where the fix lives. When adding tests:
- Prefer adding to existing test parameterizations if a related test already exists
- If no related test exists, write a test that compares cudf output to pandas (using `assert_eq` or `assert_exceptions_equal`)
- Place cudf classic tests alongside existing tests in `python/cudf/cudf/tests/`
- Place cudf.pandas-specific tests in `python/cudf/cudf_pandas_tests/`

---

## Step 6 — Commit

Stage only the intended files — never anything from `pandas-testing/`:

```bash
git add python/cudf/cudf/          # source fix (if applicable)
git add python/cudf/cudf/pandas/scripts/conftest-patch.py   # xfail removal
git status                          # verify nothing from pandas-testing/ is staged
git commit -m "fix(cudf.pandas): <short description>

Fixes the following failing pandas tests:
- tests/groupby/test_reductions.py::test_first_last_skipna[Float64-False-False-first]"
```

If many node IDs share the same root cause, summarize them in the commit message rather than listing every parametrized variant. Commit message should convey what was wrong and what was fixed.

---

## STOP Conditions

Stop immediately and report findings to the user when any of the following apply:

- The fix requires modifying `.pyx`, `.cu`, `.cuh`, or `CMakeLists.txt` files — this requires compilation, which is outside the scope of this skill
- Investigation reveals the behavioral divergence is **intentional** — cudf was deliberately designed to differ from pandas
- You have been through 3 or more investigation cycles without converging on a fix
- The fix would add a new `mode.pandas_compatible` conditional

For intentional divergence: stop and ask the user. In most cases, the goal is to make cudf match pandas. Only when the divergence has significant performance implications should `mode.pandas_compatible` be used — and even then, sparingly. cudf already overuses this flag; the default behavior should agree with pandas.

---

## Important Notes

- `mode.pandas_compatible` is automatically set to `True` when `cudf.pandas` is active. This is done at the end of `python/cudf/cudf/pandas/_wrappers/pandas.py`.
- cudf Python is almost entirely pure Python — for inplace installs, changes to `.py` files take effect immediately without rebuilding.
- `pandas-testing/pandas-tests/` must be deleted before each test run to avoid duplicate conftest hook registrations (the script appends the patch file on every run).
- Keys in all three `conftest-patch.py` dictionaries must remain in alphabetical order.
- Never write comments that explain what old code was replaced — write comments about what the code does and why.
- Never modify the pandas test files themselves — fix cudf, not pandas.
- Never fix the testing APIs (like `assert_frame_equal`, `assert_series_equal`) — fix the actual APIs that produce wrong results.
- First see if the problem is in cudf classic and fix it there; if not, then move over to cudf.pandas.
- Tests run with `xfail_strict = true` — a test listed in `NODEIDS_THAT_FAIL` that unexpectedly passes is reported as `XPASS` (also a failure). Remove from the list before testing.
