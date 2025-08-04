# LTO-IR Support for JIT Compilation in cuDF

## Overview

This feature adds experimental support for LTO-IR (Link Time Optimization Intermediate Representation) to cuDF's JIT compilation system. The goal is to significantly reduce JIT compilation time by pre-compiling common operators to LTO-IR at build time, then linking them at runtime instead of compiling from CUDA C++ source.

## Background

Current cuDF JIT compilation uses NVRTC to compile CUDA C++ code at runtime. Profiling shows that ~90% of compilation time is spent in the CUDA C++ frontend, making JIT compilation prohibitively expensive for latency-sensitive workloads.

LTO-IR compilation addresses this by:
1. Pre-compiling common operators to LTO-IR at build time
2. At runtime, only linking pre-compiled LTO-IR modules and generating final SASS
3. Falling back to traditional CUDA C++ compilation for unsupported operations

## Building with LTO-IR Support

### CMake Option

Enable LTO-IR support during build:

```bash
cmake -DCUDF_USE_LTO_IR=ON ..
```

Or using the build script:

```bash
./build.sh libcudf --cmake-args="-DCUDF_USE_LTO_IR=ON"
```

### Build-time Effects

When enabled, this option:
- Adds `--device-lto` flag to CUDA compilation
- Includes LTO-IR infrastructure in the build
- Pre-compiles common operators to LTO-IR (future work)

## Runtime Configuration

### Environment Variables

Control LTO-IR behavior at runtime with these environment variables:

- `CUDF_JIT_COMPILATION_MODE`: 
  - `auto` (default): Automatically choose based on availability
  - `lto_ir_only`: Use only LTO-IR, fail if not available
  - `cuda_only`: Use only traditional CUDA C++ compilation  
  - `prefer_lto_ir`: Prefer LTO-IR but fall back to CUDA C++

- `CUDF_JIT_AGGRESSIVE_DETECTION`: 
  - `true`: Use aggressive operator detection (may be less accurate)
  - `false` (default): Use conservative operator detection

### Example Usage

```bash
# Force LTO-IR only (for testing)
export CUDF_JIT_COMPILATION_MODE=lto_ir_only

# Prefer LTO-IR with fallback (recommended)
export CUDF_JIT_COMPILATION_MODE=prefer_lto_ir

# Disable LTO-IR completely
export CUDF_JIT_COMPILATION_MODE=cuda_only
```

## Supported Operations

Currently, LTO-IR support is being developed for:

### Binary Operations
- Arithmetic: add, subtract, multiply, divide
- Comparison: equal, not_equal, less, greater, less_equal, greater_equal  
- Logical: logical_and, logical_or

### Transform Operations
- Mathematical functions: sin, cos, exp, log, sqrt, abs

## Architecture

### Key Components

1. **LTO-IR Cache (`src/jit/lto_ir.hpp|cpp`)**:
   - Manages pre-compiled LTO-IR operators
   - Handles operator registration and retrieval
   - Links operators at runtime

2. **Configuration (`src/jit/config.hpp|cpp`)**:
   - Runtime configuration management
   - Environment variable parsing
   - Compilation mode control

3. **Integration Points**:
   - Transform operations (`src/transform/transform.cpp`)
   - Binary operations (`src/binaryop/binaryop.cpp`)

### Compilation Flow

```
User Code → Operator Detection → LTO-IR Available? 
                                      ↓ Yes
                                 Link LTO-IR → Execute
                                      ↓ No  
                                 CUDA C++ Compilation → Execute
```

## Current Status

This is an **experimental feature** with the following limitations:

1. **LTO-IR Linking**: Currently uses placeholder implementation as jitify2 LTO-IR support is pending
2. **Operator Detection**: Uses simple heuristics; more sophisticated parsing needed
3. **Pre-compiled Operators**: Framework exists but actual LTO-IR data generation is future work
4. **Testing**: Basic infrastructure tests only; performance validation needed

## Future Work

1. **Generate Actual LTO-IR**: Build system integration to pre-compile kernels to LTO-IR
2. **jitify2 Integration**: Work with jitify2 team to add LTO-IR support or use CUDA driver API directly
3. **Sophisticated Parsing**: Better operator detection from CUDA/PTX source
4. **Performance Validation**: Benchmark compilation time improvements
5. **Row-IR Integration**: Integrate with upcoming Row-IR system for AST-based compilation

## Testing

Run LTO-IR tests:

```bash
# Build with tests
./build.sh libcudf tests --cmake-args="-DCUDF_USE_LTO_IR=ON"

# Run LTO-IR specific tests
./build/tests/LTO_IR_TEST
```

## Troubleshooting

### Common Issues

1. **Build Failures**: Ensure CUDA toolkit supports `--device-lto` flag (CUDA 11.0+)
2. **Runtime Errors**: Check that `CUDF_JIT_COMPILATION_MODE` is not set to `lto_ir_only` as actual linking is not yet implemented
3. **Performance**: Current implementation may not show performance benefits until LTO-IR linking is completed

### Debug Information

Enable verbose output to see compilation decisions:

```cpp
// In application code, check what mode is being used
auto& config = cudf::jit::jit_config::instance();
std::cout << "LTO-IR enabled: " << config.is_lto_ir_enabled() << std::endl;
```