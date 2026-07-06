# Benchmarks

<!-- TODO: add PDS-DS (TPC-DS variant) section, mirroring the PDS-H instructions below -->

## PDS-H (TPC-H variant)

The steps below reproduce the PDS-H benchmark results using the Polars GPU engine.

### Setup

Install `cudf-polars` following the
[RAPIDS installation guide](https://docs.rapids.ai/install). For nightly wheels, install with
the `ray` extra (required for multi-GPU benchmarking):

```bash
CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    "cudf-polars-cu${CUDA_MAJOR}[ray]>=0.0.0a0"
```

Because `cudf-polars` pins to a tested range of Polars versions, the nightly wheel will install
the highest Polars version the GPU engine currently supports, which may not be the latest
Polars release.

Then install the benchmarks themselves. They live in the
[`cudf-benchmarks`](https://github.com/rapidsai/cudf/tree/main/python/cudf_benchmarks) package in
the cuDF repository, so install it from a checkout. The `polars` extra pulls in the CPU baselines
and `tpchgen-cli`, the Rust-based TPC-H data generator used to produce the dataset as Parquet
files:

```bash
git clone https://github.com/rapidsai/cudf.git
cd cudf
pip install -e python/cudf_benchmarks[polars]
```

#### CPU-only machines

The benchmarks also run on a machine with no GPU. The CPU frontends, `--frontend polars-cpu`
(the Polars CPU streaming engine) and `--frontend duckdb`, do not import any CUDA libraries.
On such a machine, skip the `cudf-polars` install above and install only the CPU dependencies:

```bash
git clone https://github.com/rapidsai/cudf.git
cd cudf
pip install -e python/cudf_benchmarks[cpu]
```

Then generate data and run as below, using `--frontend polars-cpu` or `--frontend duckdb`.

### Generate data

Set the scale factor once and reuse it across all steps. The following generates SF1000
(scale factor 1000, roughly 1TB of data):

```bash
export SCALE_FACTOR=1000.0
export DATA_PATH="data/tables/scale-${SCALE_FACTOR}"

tpchgen-cli --output-dir="${DATA_PATH}" --format=parquet -s ${SCALE_FACTOR}
```

### Run

**CPU** (`--frontend polars-cpu`, Polars CPU streaming engine):

```bash
python -m cudf_benchmarks.polars.pdsh all \
    --frontend polars-cpu \
    --path "${DATA_PATH}"
```

**Single GPU** (`--frontend spmd`, single-process streaming executor, equivalent to `collect(engine="gpu")`):

```bash
python -m cudf_benchmarks.polars.pdsh all \
    --frontend spmd \
    --path "${DATA_PATH}"
```

**Multi GPU** (`--frontend ray`, Ray-managed distributed streaming executor):

If running inside a Docker container, increase `/dev/shm` by passing `--shm-size=16g` to
`docker run`. All multi-GPU frontends use UCX for intra-node communication, which relies on
POSIX shared memory (`/dev/shm`) for GPU-to-GPU transfers. Docker's default `/dev/shm` is
64MB, which is far too small and will cause failures on any non-trivial workload.

By default all visible GPUs are used. To select specific devices, set `CUDA_VISIBLE_DEVICES`.
To limit the number of GPUs, use `--num-gpus`:

```bash
# All visible GPUs
python -m cudf_benchmarks.polars.pdsh all \
    --frontend ray \
    --path "${DATA_PATH}"

# Specific devices
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m cudf_benchmarks.polars.pdsh all \
    --frontend ray \
    --path "${DATA_PATH}"

# Limit to N GPUs
python -m cudf_benchmarks.polars.pdsh all \
    --frontend ray \
    --num-gpus 4 \
    --path "${DATA_PATH}"
```

### Results

Results are written to `pdsh_results.jsonl` in the current directory by default (override with `-o`).
Each run appends one JSON line containing metadata and a `records` field with per-query,
per-iteration timings:

```json
{
  "engine_name": "cudf-polars",
  "frontend": "spmd",
  "dataset_path": "data/tables/scale-1000.0",
  "scale_factor": 1000,
  "records": {
    "1": [
      {"query": 1, "iteration": 0, "duration": 0.79, "status": "success"},
      {"query": 1, "iteration": 1, "duration": 0.55, "status": "success"}
    ]
  }
}
```

`duration` is in seconds. Running multiple frontends with the same `-o` file appends each as a
separate line, making it easy to compare CPU and GPU results in one file.

### Tuning

The commands above use default settings, which gives a realistic baseline without manual tuning. The most impactful options to adjust are:

| Option | Description |
|--------|-------------|
| `--target-partition-size` | Target IO chunk size in bytes fed to the GPU. The most impactful lever; tune this first if query performance is below expectations. Default: `min(2.5% of smallest GPU memory, 1.5GB)`. |
| `--broadcast-limit` | Maximum table size in bytes for broadcast joins instead of shuffle. Increasing this can significantly speed up join-heavy queries. Default: `min(15% of smallest GPU memory, 16GB)`. |
| `--spill-device-limit` | GPU memory usage percentage before spilling to host. Lower this if hitting out-of-memory errors. Default: `80%`. |
| `--pinned-memory` / `--pinned-initial-pool-size` | Enable a pinned host memory pool for faster CPU-to-GPU transfers. Off by default. When enabled, the pool starts empty and grows up to 80% of host memory per GPU; set `--pinned-initial-pool-size` (bytes) to pre-allocate capacity upfront. |
