# Memory Resources Tests Documentation

## Overview

This directory contains comprehensive tests for the `cudf::memory_resources` class, which enables separate control of temporary and output memory allocations in libcudf.

## Test Files

### 1. `memory_resources_tests.cpp`
Primary test suite covering core functionality and real-world usage.

#### Test Categories:

**Constructor Tests**
- `TwoArgumentConstructor` - Verifies explicit output and temporary MR specification
- `SingleArgumentConstructor` - Tests backward compatibility constructor
- `ImplicitConversion` - Validates implicit conversion from `rmm::device_async_resource_ref`

**Separate Memory Pool Tests**
- `SeparateMemoryPools` - Basic test with two distinct pool resources
- `SeparatePoolsWithTracking` - Verifies allocations go to correct pools using tracking adaptors
- `StatisticsResourceAdaptor` - Tests with statistics adaptors to monitor allocation counts

**API Compatibility Tests**
- `BackwardCompatibilityWithResourceRef` - Ensures old code still works
- `DefaultParameterWorks` - Tests default parameter behavior
- `ThreadingResourcesThroughCalls` - Verifies resources are passed correctly through nested calls

**Functional Tests**
- `GatherWithSeparateResources` - Tests gather operation with separate pools
- `ColumnFactoriesUseSeparateResources` - Verifies column creation uses correct resources

**Edge Case Tests**
- `ZeroSizeColumn` - Handles zero-size allocations
- `LargeAllocation` - Tests with 1M element column
- `NullMaskAllocation` - Verifies null mask uses output resource

**Accessor Tests**
- `GettersReturnCorrectReferences` - Validates getter methods
- `GettersAreNoexcept` - Confirms no-throw guarantee

**Copy/Assignment Tests**
- `CopyConstructor` - Tests copy semantics
- `CopyAssignment` - Tests assignment operator

### 2. `memory_resources_validation_tests.cpp`
Advanced tests for validation mode and resource threading verification.

#### Test Categories:

**Validation Mode Tests**
- `ValidationModeThrowsWhenEnabled` - Environment variable enables validation
- `ValidationModeDoesNotThrowWhenDisabled` - Normal operation when disabled
- `ValidationModeErrorMessage` - Verifies helpful error messages
- `SingleArgConstructorInValidationModeThrows` - Single-arg constructor behavior in validation
- `TwoArgConstructorWorksInValidationMode` - Two-arg constructor bypasses validation check

**Resource Threading Tests**
- `VerifyResourcesThreadedThroughAPIs` - Confirms resources passed correctly
- `ComplexOperationUsesTemporaryMemory` - Validates temporary resource usage

**Default Parameter Tests**
- `DefaultParameterUsesGetCurrent` - Tests default parameter mechanism

**Resource Lifetime Tests**
- `ResourcesOutliveOperations` - Memory remains valid after resources object destroyed

**Reuse Tests**
- `ReuseResourcesAcrossOperations` - Same resources for multiple operations

**Error Handling Tests**
- `ValidResourceReferencesRequired` - Ensures valid references work correctly

## Building and Running Tests

### Build Tests
```bash
cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
ninja memory_resources_tests memory_resources_validation_tests
```

### Run All Memory Resources Tests
```bash
cd cpp/build
ctest -R memory_resources -V
```

### Run Specific Test Suite
```bash
# Run basic functionality tests
./tests/utilities_tests/memory_resources_tests

# Run validation tests
./tests/utilities_tests/memory_resources_validation_tests
```

### Run Specific Test
```bash
./tests/utilities_tests/memory_resources_tests --gtest_filter=MemoryResourcesTest.TwoArgumentConstructor
```

### Run with Validation Mode
```bash
export LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF=1
./tests/utilities_tests/memory_resources_tests
./tests/utilities_tests/memory_resources_validation_tests
unset LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF
```

## Test Coverage

### What's Tested ✅

1. **Core Functionality**
   - Both constructor variants
   - Getter methods
   - Implicit conversion
   - Copy/assignment operations

2. **Memory Resource Separation**
   - Separate pools for output vs temporary
   - Tracking allocations to verify correct resource usage
   - Statistics monitoring

3. **API Compatibility**
   - Backward compatibility with existing code
   - Default parameters
   - Implicit conversion from resource refs

4. **Real Operations**
   - Column creation
   - Gather operations
   - Column from scalar
   - Various column types (numeric, fixed-point)

5. **Validation Mode**
   - Environment variable activation
   - Error messages
   - Single vs two-argument constructor behavior

6. **Edge Cases**
   - Zero-size columns
   - Large allocations
   - Null mask allocations
   - Resource lifetime

### What's Not Tested (Future Work)

1. **Multi-threaded scenarios**
   - Thread-safe resource usage
   - Concurrent operations with separate resources

2. **All Operation Types**
   - Join operations
   - Groupby operations
   - Sort operations
   - String operations
   - List operations
   - More complex operations

3. **Performance**
   - Overhead of passing resources struct
   - Memory efficiency comparisons

4. **Stress Tests**
   - Memory exhaustion scenarios
   - Very large allocation sequences
   - Rapid allocation/deallocation

## Adding New Tests

### Template for Basic Test
```cpp
TEST_F(MemoryResourcesTest, YourTestName)
{
  auto mr = rmm::mr::get_current_device_resource();
  cudf::memory_resources resources(mr);

  auto stream = cudf::get_default_stream();

  // Your test code here
  auto result = cudf::some_operation(..., stream, resources);

  // Assertions
  EXPECT_EQ(result->size(), expected_size);
}
```

### Template for Validation Test
```cpp
TEST_F(MemoryResourcesValidationTest, YourValidationTest)
{
  setenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF", "1", 1);

  // Test code that should behave differently in validation mode

  unsetenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF");
}
```

### Template for Tracking Test
```cpp
TEST_F(MemoryResourcesTest, YourTrackingTest)
{
  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> temp_tracking{cuda_mr};

  output_tracking.reset_allocations();
  temp_tracking.reset_allocations();

  cudf::memory_resources resources(&output_tracking, &temp_tracking);

  // Perform operations
  // ...

  // Verify allocations
  EXPECT_GT(output_tracking.get_total_allocated_count(), 0);
  EXPECT_GT(temp_tracking.get_total_allocated_count(), 0);
}
```

## Debugging Failed Tests

### Common Issues

1. **Test fails with "no matching function"**
   - Check that the function signature has been updated to accept `cudf::memory_resources`
   - Verify includes are correct

2. **Validation mode test fails**
   - Ensure environment variable is set/unset correctly
   - Check for code paths that call `get_current_device_resource_ref()` directly

3. **Tracking shows unexpected allocations**
   - Some operations may use temporary memory unexpectedly
   - Check if operation is calling other functions that allocate
   - Verify the tracking adaptor is reset before the operation

### Useful Debug Techniques

```cpp
// Print allocation statistics
std::cout << "Output allocated: " << output_tracking.get_bytes_allocated() << " bytes\n";
std::cout << "Output count: " << output_tracking.get_total_allocated_count() << "\n";
std::cout << "Temp allocated: " << temp_tracking.get_bytes_allocated() << " bytes\n";
std::cout << "Temp count: " << temp_tracking.get_total_allocated_count() << "\n";
```

## Integration with CI

These tests should be run as part of the standard test suite:

```bash
# In CI pipeline
cd cpp/build
ctest --output-on-failure

# With validation mode
export LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF=1
ctest -R memory_resources --output-on-failure
unset LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF
```

## Related Documentation

- **Implementation Plan**: `/home/coder/.claude/plans/kind-chasing-blum.md`
- **Refactoring Summary**: `/home/coder/cudf/REFACTORING_SUMMARY.md`
- **Fixes Applied**: `/home/coder/cudf/FIXES_APPLIED.md`
- **Validation Script**: `/home/coder/cudf/validate_refactoring.sh`

## Contact

For questions or issues with these tests, refer to the refactoring documentation or file an issue in the cuDF repository.
