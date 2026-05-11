---
name: repro-nvcomp-race
description: Reproduce the nvCOMP/cuDF intermittent ZSTD decompression race condition in a cuDF devcontainer.
---

# Instructions
Activate when the user calls `/repro-nvcomp-race` or asks to reproduce, investigate, or debug the nvcomp ZSTD race or refers to SeanRooooney/nvcomp-zstd-reproducer.

# Goal

Instructions to reproduce the intermittent ZSTD decompression bug in cuDF when multiple `chunked_parquet_reader` instances read concurrently. Upstream report: https://github.com/SeanRooooney/nvcomp-zstd-reproducer.

## Prerequisites

- Inside a cuDF devcontainer (username is `coder`). If not, stop and ask the user to run inside cuDF devcontainer.
- NVIDIA GPU(s) accessible (verify with `nvidia-smi -L`).
- `pyarrow` available; `duckdb` will be installed if missing.

## Bug Conditions (all required)

1. Multiple concurrent `chunked_parquet_reader` instances
2. Streams from `cudf::detail::global_cuda_stream_pool()`
3. ZSTD-compressed Parquet files
4. Shared `cuda_async_memory_resource`
5. Environment variable`LIBCUDF_NVCOMP_POLICY` is set to `ALWAYS`

Note: All these are being handled in the new libcudf example called `nvcomp_zstd_repro`. Failure rate is intermittent (upstream reports ~60-70% of runs). Higher thread count, more files, and more iterations all increase trigger probability.

## Workflow

### Step 1: Clone the upstream reproducer repo

Clone to `~/` (only the `tpch_generator/` is used; the C++ reproducer is already in this cuDF branch as a libcudf example):

```bash
[ -d ~/nvcomp-zstd-reproducer ] || git clone https://github.com/SeanRooooney/nvcomp-zstd-reproducer.git ~/nvcomp-zstd-reproducer
```

Inspect `~/nvcomp-zstd-reproducer/tpch_generator/generate.py` and `~/nvcomp-zstd-reproducer/tpch_generator/requirements.txt` to confirm the generator interface and Python deps.

### Step 2: Install Python deps and generate SF1 TPC-H ZSTD data (if not already generated)

```bash
pip install 'duckdb>=0.10.0'
mkdir -p ~/data
python ~/nvcomp-zstd-reproducer/tpch_generator/generate.py ~/data/tpch_sf1 \
    --scale-factor 1 --files-per-table 100 --compression zstd
```

Expected output: 602 files, ~280 MB total across 8 tables (lineitem is the largest at 100 files / ~183 MB).

Verify ZSTD compression on one file before proceeding:

```bash
python -c "import pyarrow.parquet as pq; m = pq.ParquetFile('/home/coder/data/tpch_sf1/lineitem/00000-lineitem.parquet'); print(m.metadata.row_group(0).column(0).compression)"
# Expected: ZSTD
```

### Step 3: Build libcudf (if not already built)

Follow the standard cuDF build flow (same convention as `build-test-cudf` skill):

```bash
build-cudf-cpp -j0
```

For incremental rebuilds, `cd cpp/build/latest && ninja`.

### Step 4: Build the `nvcomp_zstd_repro` example

The reproducer is checked in at `cpp/examples/nvcomp_zstd_repro/`. It links against the already-built `libcudf` and `librmm` — no Prestissimo or other external deps needed.

First build (configures CMake):

```bash
cd /home/coder/cudf/cpp/examples && ./build.sh
```

For subsequent rebuilds after editing `reproducer.cpp`, use ninja directly:

```bash
cd /home/coder/cudf/cpp/examples/nvcomp_zstd_repro/build && ninja
```

Binary lands at `/home/coder/cudf/cpp/examples/nvcomp_zstd_repro/build/nvcomp_zstd_repro`.

### Step 5: Run the reproducer

Pick a free GPU with `nvidia-smi`, then export `CUDA_VISIBLE_DEVICES` and run.

Start small and escalate:

```bash
export CUDA_VISIBLE_DEVICES=0

# Tier 1: quick smoke
export LIBCUDF_NVCOMP_POLICY=ALWAYS && /home/coder/cudf/cpp/examples/nvcomp_zstd_repro/build/nvcomp_zstd_repro \
    /home/coder/data/tpch_sf1 --iterations 30 --threads 8

# Tier 2: if Tier 1 passes, increase pressure
export LIBCUDF_NVCOMP_POLICY=ALWAYS && /home/coder/cudf/cpp/examples/nvcomp_zstd_repro/build/nvcomp_zstd_repro \
    /home/coder/data/tpch_sf1 --iterations 50 --threads 16
```

If still passing, loop the run until failure or further increase `--threads` / `--iterations`.

**Note that** setting env-var `LIBCUDF_NVCOMP_POLICY=OFF` should make the race condition go away for the same test that was failing under the `ALWAYS` policy.

**Known reproduction:** on `umb-b200-220` the bug was reproduced with `--threads 16 --iterations 50` against full SF1 (602 files), failing around iteration ~46 with:

```
CUDF failure at: /home/coder/cudf/cpp/src/io/parquet/reader_impl_chunking_utils.cu:628: Error during decompression
```

## Failure Modes (all same underlying bug)

When reporting, capture the exact failure line. Expect one of:

1. `CUDF failure at: .../reader_impl_chunking_utils.cu:NNN: Error during decompression`
2. `CUDF failure at: .../reader_impl_preprocess.cu:NNN: Encountered page_offsets / num_columns mismatch`
3. `cudaErrorIllegalAddress an illegal memory access was encountered`
4. Process hang at ~100% GPU utilization (no progress; use `nvidia-smi` to confirm and `kill -9` to recover)

## Tuning Knobs in the Reproducer

```
--iterations N    Iterations per thread (default 100)
--threads N       Concurrent threads (default 5; alias --tables)
--chunk-limit N   chunked_parquet_reader chunk read limit in GB (default 4)
--pass-limit N    chunked_parquet_reader pass read limit in GB (default 16)
```

Tips for raising trigger probability:
- More threads (8 → 16 → 32)
- More iterations (30 → 50 → 100)
- Point at full SF1 dir (more files = more reader create/destroy churn) rather than a single table
- Drop OS file cache between runs (`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` if permitted) for cold-cache pressure.

## Notes / Caveats

- The upstream README mentions Prestissimo; that is **not** required here. The example links directly against the in-tree libcudf and librmm.
- If the build fails with RMM `any_resource` / `device_async_resource_ref` errors, the reproducer's MR setup must match the current cuDF API — see `cpp/examples/parquet_io/common_utils.cpp` (`create_memory_resource` returning `cuda::mr::any_resource<cuda::mr::device_accessible>` + `cudf::set_current_device_resource`).
- Individual Parquet files are not corrupt; reading any failing file 1000+ times single-threaded succeeds. The bug only manifests under concurrent multi-threaded access.
