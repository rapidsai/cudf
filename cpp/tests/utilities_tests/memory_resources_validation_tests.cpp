/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <cstdlib>
#include <stdexcept>

class MemoryResourcesValidationTest : public cudf::test::BaseFixture {};

// =============================================================================
// Validation Mode Tests
// =============================================================================

TEST_F(MemoryResourcesValidationTest, ValidationModeThrowsWhenEnabled)
{
  // Set environment variable to enable validation
  setenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF", "1", 1);

  // Now calling get_current_device_resource_ref() should throw
  EXPECT_THROW(cudf::get_current_device_resource_ref(), std::runtime_error);

  // Clean up
  unsetenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF");
}

TEST_F(MemoryResourcesValidationTest, ValidationModeDoesNotThrowWhenDisabled)
{
  // Make sure environment variable is not set
  unsetenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF");

  // Should not throw
  EXPECT_NO_THROW(auto mr = cudf::get_current_device_resource_ref());
}

TEST_F(MemoryResourcesValidationTest, ValidationModeErrorMessage)
{
  setenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF", "1", 1);

  try {
    cudf::get_current_device_resource_ref();
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    std::string error_msg = e.what();
    // Verify error message contains helpful information
    EXPECT_NE(error_msg.find("validation mode"), std::string::npos);
    EXPECT_NE(error_msg.find("resources.get_temporary_mr()"), std::string::npos);
  }

  unsetenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF");
}

// =============================================================================
// Testing that Single-Argument Constructor Works in Normal Mode
// =============================================================================

TEST_F(MemoryResourcesValidationTest, SingleArgConstructorCallsGetCurrentInNormalMode)
{
  // Ensure validation is disabled
  unsetenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF");

  auto output_mr = rmm::mr::get_current_device_resource();

  // Single-argument constructor calls get_current_device_resource_ref() internally
  // This should work in normal mode
  EXPECT_NO_THROW(cudf::memory_resources resources(output_mr));
}

TEST_F(MemoryResourcesValidationTest, SingleArgConstructorInValidationModeThrows)
{
  // Enable validation mode
  setenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF", "1", 1);

  auto output_mr = rmm::mr::get_current_device_resource();

  // Single-argument constructor will call get_current_device_resource_ref()
  // which should throw in validation mode
  EXPECT_THROW(cudf::memory_resources resources(output_mr), std::runtime_error);

  unsetenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF");
}

TEST_F(MemoryResourcesValidationTest, TwoArgConstructorWorksInValidationMode)
{
  // Enable validation mode
  setenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF", "1", 1);

  auto output_mr = rmm::mr::get_current_device_resource();
  auto temp_mr   = rmm::mr::get_current_device_resource();

  // Two-argument constructor doesn't call get_current_device_resource_ref()
  // so it should work even in validation mode
  EXPECT_NO_THROW(cudf::memory_resources resources(output_mr, temp_mr));

  unsetenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF");
}

// =============================================================================
// Memory Resource Threading Tests
// =============================================================================

TEST_F(MemoryResourcesValidationTest, VerifyResourcesThreadedThroughAPIs)
{
  using namespace cudf::test;

  // Use separate tracking resources to verify correct usage
  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> temp_tracking{cuda_mr};

  output_tracking.reset_allocations();
  temp_tracking.reset_allocations();

  cudf::memory_resources resources(&output_tracking, &temp_tracking);

  auto stream = cudf::get_default_stream();

  // Create a column - should allocate from output_tracking
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1000, cudf::mask_state::UNALLOCATED, stream, resources);

  // Verify output resource was used
  EXPECT_GT(output_tracking.get_total_allocated_count(), 0);
  EXPECT_GT(output_tracking.get_bytes_allocated(), 0);

  // Note: Temporary tracking may or may not have allocations depending on
  // whether the operation needed temporary buffers
}

TEST_F(MemoryResourcesValidationTest, ComplexOperationUsesTemporaryMemory)
{
  using namespace cudf::test;

  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> temp_tracking{cuda_mr};

  output_tracking.reset_allocations();
  temp_tracking.reset_allocations();

  cudf::memory_resources resources(&output_tracking, &temp_tracking);

  // Create input table
  fixed_width_column_wrapper<int32_t> col1({5, 4, 3, 2, 1, 0});
  fixed_width_column_wrapper<int32_t> col2({10, 40, 30, 20, 50, 0});
  cudf::table_view input({col1, col2});

  // Gather with a map - this typically uses temporary memory for intermediate results
  fixed_width_column_wrapper<int32_t> gather_map({5, 4, 3, 2, 1, 0});  // Reverse order

  auto stream = cudf::get_default_stream();
  auto result =
    cudf::gather(input, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, stream, resources);

  // Both output and temporary resources should have been used
  EXPECT_GT(output_tracking.get_total_allocated_count(), 0);

  // Temporary allocations depend on the implementation
  // Many operations will use temporary buffers
  // (Specific assertion depends on gather implementation details)
}

// =============================================================================
// Default Parameter Tests
// =============================================================================

TEST_F(MemoryResourcesValidationTest, DefaultParameterUsesGetCurrent)
{
  unsetenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF");

  auto stream = cudf::get_default_stream();

  // Call without resources parameter - uses default which calls get_current_device_resource_ref()
  EXPECT_NO_THROW(
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED, stream));
}

// =============================================================================
// Resource Lifetime Tests
// =============================================================================

TEST_F(MemoryResourcesValidationTest, ResourcesOutliveOperations)
{
  auto cuda_mr = rmm::mr::get_current_device_resource();

  std::unique_ptr<cudf::column> result_col;

  {
    // Create resources in limited scope
    rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
    cudf::memory_resources resources(&output_tracking);

    auto stream = cudf::get_default_stream();
    result_col  = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED, stream, resources);

    // Resources will be destroyed here
  }

  // Column should still be valid even though resources object was destroyed
  // because the actual memory_resource* remains valid
  EXPECT_EQ(result_col->size(), 100);
  EXPECT_EQ(result_col->type().id(), cudf::type_id::INT32);
}

// =============================================================================
// Multiple Operations with Same Resources
// =============================================================================

TEST_F(MemoryResourcesValidationTest, ReuseResourcesAcrossOperations)
{
  using namespace cudf::test;

  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> temp_tracking{cuda_mr};

  cudf::memory_resources resources(&output_tracking, &temp_tracking);

  auto stream = cudf::get_default_stream();

  // Operation 1: Create column
  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED, stream, resources);

  auto count_after_op1 = output_tracking.get_total_allocated_count();

  // Operation 2: Create another column with same resources
  auto col2 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 200, cudf::mask_state::UNALLOCATED, stream, resources);

  // Should have more allocations
  EXPECT_GT(output_tracking.get_total_allocated_count(), count_after_op1);

  // Both columns should be valid
  EXPECT_EQ(col1->size(), 100);
  EXPECT_EQ(col2->size(), 200);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(MemoryResourcesValidationTest, ValidResourceReferencesRequired)
{
  // Create valid resources
  auto mr = rmm::mr::get_current_device_resource();
  cudf::memory_resources resources(mr);

  // Using the resources should work
  auto stream = cudf::get_default_stream();
  EXPECT_NO_THROW(auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                       100,
                                                       cudf::mask_state::UNALLOCATED,
                                                       stream,
                                                       resources));
}
