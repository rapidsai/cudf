# Run Baseline CSV Benchmark

Run the baseline benchmark for the CSV parser without making any code changes.

Target: $ARGUMENTS (default: csv)

Steps:
1. Build: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
2. Check build: `tail -n 20 build.log`
3. Run tests: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
4. Check tests: `grep -c "FAILED\|PASSED" test.log`
5. Run CSV reader benchmark: `./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > reader_run.log 2>&1`
6. Run CSV writer benchmark: `./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > writer_run.log 2>&1`
7. Extract metrics: `grep -E "Elem/s|Bytes/s|GlobalMem BW|BWUtil|time" reader_run.log writer_run.log`
8. Display results
9. Clean up: `rm -f build.log test.log reader_run.log writer_run.log`
