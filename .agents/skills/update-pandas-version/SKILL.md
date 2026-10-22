---
name: update-pandas-version
description: Use when bumping the supported pandas version in cudf (a new minor or major pandas release), including the pins in dependencies.yaml and the test skip and xfail lists used by CI
---

# Updating the supported pandas version in cudf

This skill describes the procedure for bumping the pandas version that cudf supports. A successful update changes the pandas pins, then verifies locally that the pandas-related CI test suites pass with the target pandas version, recording any new failures in the lists that CI uses.

cudf supports every minor and patch version of pandas within one major version. The bump type therefore decides which pin bound changes:

- **Minor version bump** (for example 3.0 to 3.1): change only the upper bound. `pandas>=3.0.0,<3.1.0` becomes `pandas>=3.0.0,<3.2.0`.
- **Major version bump** (for example 3.x to 4.0): change both bounds. `pandas>=3.0.0,<3.2.0` becomes `pandas>=4.0.0,<4.1.0`.

## Prerequisites
1. Prompt the user for the target pandas version for the upgrade.
2. Read the current `&pandas` pin in the `run_common` section of `dependencies.yaml` and decide whether the target is a minor or a major version bump.

## Step 1. Update the pandas pins in `dependencies.yaml`

1. Change the `&pandas` anchor in the `run_common` section as described above for a minor or major version bump. The `*pandas` aliases reuse this pin.
2. Update the `test_cudf_pandas_compat` matrix so it has one entry for each supported pandas minor version. Each entry sets `pandas_compat_version: "X.Y"` and pins `pandas==X.Y.*`, together with the same `numpy` pin as the existing entry. Keep the trailing `- matrix:` entry with no packages. For a major version bump, remove the entries of the old major version, so only the new `X.0` entry is left.
3. Search `dependencies.yaml` and `python/cudf/cudf_pandas_tests/third_party_integration_tests/dependencies.yaml` for pins whose comments mention a pandas version, release, or pull request. If the target pandas version contains the fix, remove the pin and its comment. If you can't tell, keep the pin and report it to the user.
4. Regenerate the files derived from `dependencies.yaml`:
   ```bash
   pre-commit run rapids-dependency-file-generator --all-files
   ```

## Step 2. Create a conda environment for testing

Run every later step in this environment.

1. Choose the file in `conda/environments/` that matches the machine. The files are named `all_cuda-<CUDA version>_arch-<architecture>.yaml`. Use the architecture that `uname -m` prints, and the newest CUDA version that is not newer than the CUDA version that `nvidia-smi` prints. Step 1 regenerated these files, so they contain the new pandas pin.
2. Create the environment. If an environment with the same name already exists, ask the user before you replace it.
   ```bash
   conda env create --yes -n cudf-pandas-3.1 -f conda/environments/all_cuda-133_arch-x86_64.yaml
   ```
3. Build cudf and dask_cudf into the environment from the repository root. A full libcudf build takes a long time, so run it in the background and wait for it to finish.
   ```bash
   conda activate cudf-pandas-3.1
   PARALLEL_LEVEL=$(nproc) ./build.sh libcudf pylibcudf cudf dask_cudf --pydevelop
   ```
4. Confirm that the packages import and that the environment has the target pandas version:
   ```bash
   python -c "import cudf, dask_cudf, pandas; print(pandas.__version__)"
   ```
   If it prints a different pandas version, check the pandas pin in the environment file from Step 1, and report the mismatch to the user before you continue.

## Step 3. Iterate and classify failures in the pandas test suite under cudf.pandas

The `pandas-tests` CI job (`ci/cudf_pandas_scripts/pandas-tests/run.sh`) runs the pandas test suite of the installed pandas version with cudf.pandas through `python/cudf/cudf/pandas/scripts/run-pandas-tests.sh`. The script clones the pandas tag `v<installed version>` into `pandas-testing/`, but only when that directory is missing. Delete the whole `pandas-testing/` directory before the first run with the new version.

The plugin `python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py` marks tests before they run. Its dictionaries map a pytest node id (`tests/...py::<test>[...]`) to a reason:

- `NODEIDS_THAT_FAIL`: marked `xfail`. Use for tests that fail because of a cudf or cudf.pandas bug. Write the root cause as the reason, not only the exception type.
- `NODEIDS_THAT_MAY_FAIL`: marked `xfail(strict=False)`. Use for flaky or environment-sensitive tests.
- `NODEIDS_TO_SKIP`: marked `skip`. Use for tests that crash, hang, or exercise behavior that cudf.pandas cannot support.
- `NODEIDS_TO_SKIP_WHEN_SHARDED`: skipped only in sharded CI runs. Use for tests that fail only when the suite is split across runners.
- `NODEIDS_PATHS_TO_SKIP`: skips every test whose node id starts with the key.

Keep the keys of every dictionary in alphabetical order. The vendored `pandas-testing/pandas-tests/pyproject.toml` sets `xfail_strict = true`, so a `NODEIDS_THAT_FAIL` entry that now passes reports `[XPASS(strict)]` and fails the run. Remove such entries.

1. Run the suite from the repository root with the same arguments as CI, without the sharding and JUnit arguments, and keep the output:
   ```bash
   bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
     --numprocesses 8 --dist worksteal --tb=line --max-worker-restart=3 \
     --report-log=pr.json 2>&1 | tee pandas-tests-output.txt
   ```
   The script moves `pr.json` to `pandas-testing/pr.json`. It runs from `pandas-testing/pandas-tests/`, so the node ids in the report log have the same format as the dictionary keys.
2. Remove dictionary entries whose node id no longer exists in the new pandas test suite. To list the current node ids, run `python -m pytest -p cudf.pandas -p cudf.pandas.scripts.pandas-testing-plugin --import-mode=importlib --collect-only -q` from `pandas-testing/pandas-tests/`. Do not add the `-m` filter of `run-pandas-tests.sh`: it hides tests that still exist.
3. List the failing tests:
   ```bash
   jq -r 'select(."$report_type" == "TestReport" and .outcome == "failed") | .nodeid' pandas-testing/pr.json | sort -u
   ```
4. For each failing test, add or remove a dictionary entry as described above. To confirm that a failure is caused by cudf.pandas, run the test without `-p cudf.pandas`: `python -m pytest pandas-testing/pandas-tests/<node id>`. If the test also fails with plain pandas because a package is missing, add the package to `test_python_pandas_tests` in `dependencies.yaml` instead.
5. Rerun only the tests you changed, then repeat the full run until it reports no failures.

## Step 4. Run the cudf and dask_cudf test suites

cudf tests compare cudf results with pandas results, so a pandas behavior change shows up as a test failure. Run the same suites as `ci/test_python_cudf.sh` and `ci/test_wheel_dask_cudf.sh`:

```bash
./ci/run_cudf_pytests.sh --numprocesses=8 --dist=worksteal
./ci/run_cudf_pandas_pytest_benchmarks.sh --benchmark-disable --numprocesses=8 --dist=worksteal
./ci/run_dask_cudf_pytests.sh --numprocesses=8 --dist=worksteal
```

## Step 5. Run the cudf.pandas unit tests

Run the same `pytest` invocations as the start of `ci/cudf_pandas_scripts/run_tests.sh`, without the `--cov` arguments, from the repository root. The profiler tests fail under pytest-xdist, so they run separately without parallel workers.

```bash
export JUPYTER_PLATFORM_DIRS=1

# Parallel tests
python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    --numprocesses=8 \
    --dist=worksteal \
    -k "not profiler" \
    -m "not serial" \
    --config-file=./python/cudf/pyproject.toml \
    ./python/cudf/cudf_pandas_tests/

# Serial tests
python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    -k "not profiler" \
    -m "serial" \
    --config-file=./python/cudf/pyproject.toml \
    ./python/cudf/cudf_pandas_tests/

# Profiler tests
python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    --config-file=./python/cudf/pyproject.toml \
    --numprocesses=0 \
    -k "profiler" \
    ./python/cudf/cudf_pandas_tests/
```

If notebook tests fail because no `python3` Jupyter kernel exists, run `python -m ipykernel install --user --name python3` as CI does, then rerun them.

## Step 6. Run the Narwhals tests with cudf.pandas

Run the cudf.pandas variation of `ci/test_narwhals.sh` against the target pandas version, keeping the current Narwhals version. Record failures in `ci/narwhals_cudf_pandas_test_plugin.py` as described in Step 3 of the `update-narwhals-testing` skill.

## Scope and report

Do **not** fix cudf, cudf.pandas, or their tests to make a failure pass. This is out of scope.

When every step is done, report:

- The pin changes, the new `test_cudf_pandas_compat` matrix, and the pins you removed or kept in Step 1.
- For each test suite, the new failures and the dictionary entries you added or removed.
- Failures in Steps 4 and 5 that you did not record anywhere.
