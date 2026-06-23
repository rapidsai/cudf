# Benchmarks

## PDS-H (TPC-H variant)

The steps below reproduce the PDS-H benchmark results using the Polars GPU engine.
No cudf source checkout is required — the benchmark script is included in the
installed `cudf-polars` package.

### Setup

Create and activate a fresh virtual environment (conda, mamba, or Python venv), then install the dependencies:

```bash
CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
pip install tpchgen-cli
pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple "cudf-polars-cu${CUDA_MAJOR}>=0.0.0a0"
```

This installs:

- **`tpchgen-cli`** — a Rust-based TPC-H data generator used to produce the benchmark dataset as Parquet files.
- **`cudf-polars`** — the Polars GPU engine, along with its dependencies including `polars`. Because the GPU engine pins to a tested range of Polars versions, the nightly wheel will install the highest Polars version that the GPU engine currently supports, which may not be the latest Polars release.

### Generate data

```bash
tpchgen-cli --output-dir="data/tables/scale-1000.0" --format=parquet -s 1000.0
```

### Run

```bash
# --frontend options: in-memory (single GPU), polars-cpu (CPU only), dask (multi-GPU)
python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend in-memory \
    --path data/tables/scale-1000.0
```

The `--path` value must match the `--output-dir` used during data generation.
Update both consistently when changing scale factors (e.g. `scale-100.0`).
