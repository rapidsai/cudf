# Run Baseline CSV Benchmark

Run the baseline benchmark for the CSV reader without making any code changes.

Target: $ARGUMENTS (default: csv)

Steps:
1. Build: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
2. Check build: `tail -n 20 build.log`
3. Run tests: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
4. Check tests: `grep -c "FAILED\|PASSED" test.log`
5. Run primary eval: `./eval.sh results/baseline`
6. Display JSON results from `results/baseline/` and NVTX stages from `results/baseline/nvtx_stages.txt`
7. Clean up: `rm -f build.log test.log`
