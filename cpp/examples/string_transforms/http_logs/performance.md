# HTTP log transform performance

Measured on 2026-07-05 with GPU 1 of a dual NVIDIA RTX A6000 system. The GPU was idle before the run. The executable was built in Release mode with CUDA 13.2 and run against the current `multi-string-output` working tree at `2e2a13526f` plus the standalone-build fixes in that tree.

Each result processes 1,000,000 rows. It is the mean of five independent process runs; each process records one cold call and the mean of ten warm calls. JIT calls use a fresh kernel-cache directory in every process. The `±` values are sample standard deviations.

| Operation | Variant | Cold (s) | Warm (ms) | Throughput (M rows/s) | Effective bandwidth (GiB/s) | Peak allocation (MiB) | Allocated/call (MiB) | Warm speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `request-line` | precompiled | 0.066313 ± 0.000504 | 68.416 ± 0.167 | 14.617 ± 0.035 | 0.975 ± 0.002 | 32.00 | 32.23 | 1.00× |
| `request-line` | runtime JIT | 1.398848 ± 0.008472 | 2.574 ± 0.017 | 388.528 ± 2.554 | 25.910 ± 0.170 | 43.32 | 54.77 | 26.58× |
| `request-line` | AOT LTO JIT-linked | 1.417205 ± 0.008892 | 2.618 ± 0.041 | 381.982 ± 5.925 | 25.473 ± 0.395 | 43.32 | 54.77 | 26.13× |
| `combined-log` | precompiled | 0.601166 ± 0.011457 | 600.868 ± 3.285 | 1.664 ± 0.009 | 0.495 ± 0.003 | 145.13 | 145.85 | 1.00× |
| `combined-log` | runtime JIT | 1.554661 ± 0.013899 | 25.394 ± 0.209 | 39.382 ± 0.323 | 11.711 ± 0.096 | 171.72 | 198.42 | 23.66× |
| `combined-log` | AOT LTO JIT-linked | 1.620733 ± 0.055994 | 25.465 ± 0.166 | 39.270 ± 0.254 | 11.677 ± 0.076 | 171.72 | 198.42 | 23.60× |

Effective bandwidth is `(input bytes + output bytes) / warm time`. Peak allocation and allocated-per-call are reported by the example's tracking memory resource; they are allocation metrics, not total device-resident memory. The precompiled implementation intentionally uses public non-JIT cuDF string and regex functions, while the JIT variants fuse parsing, sizing, and output construction.

The machine-readable comparison is in [`performance.csv`](performance.csv).
