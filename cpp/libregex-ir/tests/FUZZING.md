# Fuzzing

The default CTest suite runs `regex_ir_fuzz_smoke`, a deterministic campaign
covering arbitrary parser inputs, exhaustive small expressions, optimizer
differentials, all operations, and invalid-IR mutations.

Both the deterministic driver and libFuzzer entry point live in
`tests/fuzz_tests.cpp`; the build selects the appropriate entry point.

The flat test directory also contains `tests/fuzz_seed.hex`, a minimal seed for
the deterministic hexadecimal-input harness.

For open-ended sanitizer fuzzing, configure with Clang:

    cmake -S . -B build-fuzz -G Ninja \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DREGEX_IR_BUILD_FUZZERS=ON \
      -DREGEX_IR_BUILD_TESTS=OFF \
      -DREGEX_IR_BUILD_EXPLORER=OFF \
      -DREGEX_IR_BUILD_EXAMPLES=OFF
    cmake --build build-fuzz --target regex_ir_compile_execute_fuzzer
    ./build-fuzz/regex_ir_compile_execute_fuzzer -max_len=4096

libFuzzer controls the operation and compile options with the first input byte.
The remaining payload is split at its first newline into pattern and input.
