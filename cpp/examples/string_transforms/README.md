# libcudf C++ examples using string transforms

This C++ example demonstrates using libcudf transform API to access and create
strings columns.

Blog post that uses this code: https://developer.nvidia.com/blog/efficient-transforms-in-cudf-using-jit-compilation/

The example source code loads a csv file and produces a transformed column from the table using the values from the tables.

The following examples are included:
1. `localize_phone_jit` - Using a transform to branch on input columns and returning string values
2. `localize_phone_precompiled` - Performs same transformation on the table as `branching` but uses precompiled public APIs
3. `compute_checksum_jit` - Using a transform to perform a fused checksum on two columns
4. `extract_email_jit` - Using a transform to get a substring from a kernel
5. `extract_email_precompiled` - Performs same transformation on the table as `output` but uses precompiled public APIs
6. `format_phone_jit` - Using a transform kernel to output a string to a pre-allocated buffer
7. `format_phone_precompiled` - Performs same transformation on the table as `preallocated` but uses precompiled public APIs
8. `http_log_transforms` - Compares three multi-output HTTP log extractors:
   - `precompiled`: `cudf::strings::extract` with a public regex program.
   - `jit`: two CUDA source transforms compiled at runtime. The first produces exact per-row string
     sizes; inclusive scans turn those sizes into run-end offsets, and the second writes directly to
     the resulting string character buffers.
   - `lto`: the same sizing and output transform ABI, AOT-compiled to embedded fatbins and JIT-linked
     with libcudf's precompiled transform kernels.

The HTTP example has a medium request-line workload (method, path, and HTTP version) and a
high-complexity combined-log workload (client IP, timestamp, method, path, status, referer, and user
agent). Both implement the same extraction groups as their comparative regex variant.

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/output info.csv output.csv 100000
```

Run the HTTP example directly:

```bash
build/http_log_transforms http_logs.csv output.csv jit medium 1000000 10
build/http_log_transforms http_logs.csv output.csv lto high 1000000 10
build/http_log_transforms http_logs.csv output.csv precompiled high 1000000 10
```

Use `-` for the output path to skip CSV materialization during benchmark-only runs.

## Benchmarking and profiling

The benchmark runner records cold and warm wall time, throughput, effective input/output bandwidth,
and RMM allocation cost for all three variants. With `--profile`, Nsight Compute also records kernel
time, achieved warp occupancy, DRAM utilization, and bytes read/written. Use an otherwise idle GPU:

```bash
python tools/benchmark.py \
  --executable build/http_log_transforms \
  --input http_logs.csv \
  --output-dir results \
  --rows 100000,1000000,10000000 \
  --iterations 10 --repeats 5 --gpu 0 --profile

python tools/plot_results.py results/benchmark_results.csv
```

This creates CSV and Markdown tables plus 16:9 PNG presentation graphs using NVIDIA green, black,
and dark gray. Results should always be reported with the GPU, driver, CUDA toolkit, branch SHA, row
counts, repetitions, and exact command; the repository does not ship fabricated baseline numbers.
The runner gives every repetition a fresh `LIBCUDF_KERNEL_CACHE_PATH`, so the reported cold time
includes CUDA-source compilation or LTO linking while warm iterations reuse the in-process cache;
it also sets `LIBCUDF_JIT_DISABLE_CUDA_CACHE=1` to prevent the CUDA disk cache from hiding cold cost.

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
