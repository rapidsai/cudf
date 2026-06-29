# Benchmarks

<!-- TODO: add PDS-DS (TPC-DS variant) section, mirroring the PDS-H instructions below -->

## PDS-H (TPC-H variant)

The steps below reproduce the PDS-H benchmark results using the Polars GPU engine.

### Setup

Create and activate a fresh environment ([conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html), [Python venv](https://docs.python.org/3/library/venv.html), or [uv](https://docs.astral.sh/uv/)), then install the dependencies:

```bash
CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
pip install tpchgen-cli
pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple "cudf-polars-cu${CUDA_MAJOR}>=0.0.0a0"
```

This installs:

- **`tpchgen-cli`**: a Rust-based TPC-H data generator used to produce the benchmark dataset as Parquet files.
- **`cudf-polars`**: the Polars GPU engine, along with its dependencies including `polars`. Because the GPU engine pins to a tested range of Polars versions, the nightly wheel will install the highest Polars version that the GPU engine currently supports, which may not be the latest Polars release.

### Generate data

```bash
tpchgen-cli --output-dir="data/tables/scale-1000.0" --format=parquet -s 1000.0
```

### Run

Set these environment variables before running to match the configuration used for the published results:

```bash
export POLARS_MAX_THREADS=1
export OMP_NUM_THREADS=1
export LIBCUDF_NUM_HOST_WORKERS=4
export KVIKIO_NTHREADS=8
export RAPIDSMPF_num_streaming_threads=8
```

**Single GPU** (`--frontend spmd`, single-process streaming executor, equivalent to `collect(engine="gpu")`):

```bash
python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend spmd \
    --path data/tables/scale-1000.0
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
python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend ray \
    --path data/tables/scale-1000.0

# Specific devices
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend ray \
    --path data/tables/scale-1000.0

# Limit to N GPUs
python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend ray \
    --num-gpus 4 \
    --path data/tables/scale-1000.0
```

The `--path` value must match the `--output-dir` used during data generation.
Update both consistently when changing scale factors (e.g. `scale-100.0`).

### Tuning

The commands above use default settings, which gives a realistic baseline without manual tuning. The most impactful options to adjust are:

| Option | Description |
|--------|-------------|
| `--target-partition-size` | Target IO chunk size in bytes fed to the GPU. The most impactful lever; tune this first if query performance is below expectations. Default: auto. |
| `--broadcast-limit` | Maximum table size in bytes for broadcast joins instead of shuffle. Increasing this can significantly speed up join-heavy queries. Default: auto. |
| `--spill-device-limit` | GPU memory usage percentage before spilling to host. Lower this if hitting out-of-memory errors. Default: 80%. |
| `--pinned-memory` / `--pinned-max-pool-size` | Enable and size a pinned host memory pool for faster CPU to GPU transfers. |
