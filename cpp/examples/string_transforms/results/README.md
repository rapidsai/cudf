# Benchmark result artifacts

This directory is intentionally source-only. Run `tools/benchmark.py` on a GPU system to generate:

- `benchmark_results.csv`: raw repetitions for all variants, workloads, and row counts.
- `benchmark_results.md`: presentation-ready aggregate table.
- `ncu_<workload>_<variant>.csv`: per-kernel Nsight Compute data when `--profile` is enabled.
- `profile_results.{csv,md}`: aggregate kernel time, warp occupancy, DRAM utilization, and traffic.
- `environment.md`: branch SHA, GPU/driver, CUDA toolkit, row counts, repetitions, and profile mode.

Then run `tools/plot_results.py results/benchmark_results.csv` to create NVIDIA-colour PNG graphs
under `results/graphs/`.

Measured results are not checked in unless they identify the GPU, driver, CUDA toolkit, branch SHA,
row counts, repetitions, and exact profiling command. This avoids presenting synthetic or stale
numbers as measured performance.
