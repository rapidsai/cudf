# LTO-IR Example Usage

This example shows how to use the new LTO-IR feature in cuDF for improved JIT compilation performance.

## Building with LTO-IR Support

```bash
# Build cuDF with LTO-IR enabled
./build.sh libcudf --cmake-args="-DCUDF_USE_LTO_IR=ON"

# Or with CMake directly
mkdir build && cd build
cmake .. -DCUDF_USE_LTO_IR=ON
make -j$(nproc)
```

## Basic Usage Example

```cpp
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf_test/column_wrapper.hpp>

#include "jit/config.hpp"  // For configuration

int main() {
    // Configure LTO-IR compilation mode
    auto& config = cudf::jit::jit_config::instance();
    config.set_compilation_mode(cudf::jit::jit_config::compilation_mode::PREFER_LTO_IR);
    
    // Create test data
    cudf::test::fixed_width_column_wrapper<int32_t> lhs({1, 2, 3, 4, 5});
    cudf::test::fixed_width_column_wrapper<int32_t> rhs({10, 20, 30, 40, 50});
    
    // This will attempt LTO-IR compilation first, then fall back to CUDA C++
    auto result = cudf::binary_operation(
        lhs, rhs, 
        cudf::binary_operator::ADD, 
        cudf::data_type{cudf::type_id::INT32}
    );
    
    // Result: [11, 22, 33, 44, 55]
    std::cout << "Binary operation completed successfully" << std::endl;
    
    return 0;
}
```

## Configuration Options

### Environment Variables

```bash
# Use only LTO-IR (will fail if not available)
export CUDF_JIT_COMPILATION_MODE=lto_ir_only

# Prefer LTO-IR but allow fallback (recommended)
export CUDF_JIT_COMPILATION_MODE=prefer_lto_ir

# Use only traditional CUDA C++ compilation
export CUDF_JIT_COMPILATION_MODE=cuda_only

# Automatic mode (default)
export CUDF_JIT_COMPILATION_MODE=auto

# Enable aggressive operator detection
export CUDF_JIT_AGGRESSIVE_DETECTION=true
```

### Programmatic Configuration

```cpp
#include "jit/config.hpp"

// Get configuration instance
auto& config = cudf::jit::jit_config::instance();

// Set compilation mode
config.set_compilation_mode(cudf::jit::jit_config::compilation_mode::PREFER_LTO_IR);

// Enable aggressive operator detection
config.set_aggressive_operator_detection(true);

// Check current settings
std::cout << "LTO-IR enabled: " << config.is_lto_ir_enabled() << std::endl;
std::cout << "CUDA fallback allowed: " << config.is_cuda_fallback_allowed() << std::endl;
```

## Transform Operations Example

```cpp
#include <cudf/transform.hpp>

// Custom transform using mathematical operations
std::string cuda_source = R"(
    __device__ inline void GENERIC_TRANSFORM_OP(float& output, float input) {
        // This will be detected as using sin, multiply, and add operators
        output = sin(input) * 2.0f + 1.0f;
    }
)";

cudf::test::fixed_width_column_wrapper<float> input({0.0, 1.57, 3.14});

// LTO-IR system will detect sin, multiply, add operators and attempt LTO-IR compilation
auto result = cudf::transform(
    input, 
    cuda_source, 
    cudf::data_type{cudf::type_id::FLOAT32}, 
    false  // not PTX
);
```

## Checking LTO-IR Availability

```cpp
#include "jit/lto_ir.hpp"

#ifdef CUDF_USE_LTO_IR
auto& cache = cudf::jit::lto_ir_cache::instance();

// Check if specific operators are available
bool add_available = cache.is_lto_ir_available("binary_op", "add");
bool sin_available = cache.is_lto_ir_available("transform", "sin");

std::cout << "Add operator LTO-IR available: " << add_available << std::endl;
std::cout << "Sin operator LTO-IR available: " << sin_available << std::endl;

// Register custom LTO-IR operator
std::vector<std::string> custom_lto_ir = {"custom LTO-IR data"};
cache.register_operator("custom::my_op", custom_lto_ir);
#endif
```

## Performance Comparison

```cpp
#include <chrono>

void benchmark_compilation_modes() {
    auto& config = cudf::jit::jit_config::instance();
    
    // Test data
    cudf::test::fixed_width_column_wrapper<float> lhs(1000, 1.0f);
    cudf::test::fixed_width_column_wrapper<float> rhs(1000, 2.0f);
    
    // Benchmark LTO-IR mode
    config.set_compilation_mode(cudf::jit::jit_config::compilation_mode::PREFER_LTO_IR);
    auto start = std::chrono::high_resolution_clock::now();
    
    auto result1 = cudf::binary_operation(
        lhs, rhs, cudf::binary_operator::ADD, cudf::data_type{cudf::type_id::FLOAT32});
    
    auto lto_ir_time = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark CUDA C++ mode
    config.set_compilation_mode(cudf::jit::jit_config::compilation_mode::CUDA_ONLY);
    start = std::chrono::high_resolution_clock::now();
    
    auto result2 = cudf::binary_operation(
        lhs, rhs, cudf::binary_operator::ADD, cudf::data_type{cudf::type_id::FLOAT32});
    
    auto cuda_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "LTO-IR compilation time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(lto_ir_time).count() 
              << " μs" << std::endl;
    std::cout << "CUDA C++ compilation time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(cuda_time).count() 
              << " μs" << std::endl;
}
```

## Error Handling

```cpp
try {
    auto& config = cudf::jit::jit_config::instance();
    config.set_compilation_mode(cudf::jit::jit_config::compilation_mode::LTO_IR_ONLY);
    
    // This may throw if LTO-IR is not available for the requested operation
    auto result = cudf::binary_operation(
        lhs, rhs, cudf::binary_operator::COMPLEX_CUSTOM_OP, // Not available in LTO-IR
        cudf::data_type{cudf::type_id::INT32}
    );
} catch (const cudf::logic_error& e) {
    std::cerr << "LTO-IR compilation failed: " << e.what() << std::endl;
    
    // Fall back to allowing CUDA C++ compilation
    auto& config = cudf::jit::jit_config::instance();
    config.set_compilation_mode(cudf::jit::jit_config::compilation_mode::AUTO);
}
```

## Compilation Tips

1. **Build Flags**: Make sure to use `--device-lto` when building CUDA code that will be used for LTO-IR
2. **Architecture**: LTO-IR works best with consistent target architectures
3. **Debugging**: Use verbose compilation modes to see which path (LTO-IR vs CUDA C++) is being taken
4. **Performance**: Initial LTO-IR compilation may not show benefits until the cache is warmed up

## Current Limitations

- LTO-IR linking is not yet fully implemented (placeholder)
- Operator detection uses simple heuristics
- No actual LTO-IR data is generated yet (future work)
- Performance benefits are theoretical until implementation is complete

## Future Improvements

- Integration with jitify2 LTO-IR support
- Sophisticated operator parsing from CUDA/PTX source
- Build-time generation of actual LTO-IR data
- Performance benchmarking and optimization