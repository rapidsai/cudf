/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>

#include <memory>

class MemoryResourcesTest : public cudf::test::BaseFixture {};

// =============================================================================
// Constructor Tests
// =============================================================================

TEST_F(MemoryResourcesTest, TwoArgumentConstructor)
{
  auto output_mr = rmm::mr::get_current_device_resource();
  auto temp_mr   = rmm::mr::get_current_device_resource();

  cudf::memory_resources resources(output_mr, temp_mr);

  EXPECT_EQ(resources.get_output_mr(), output_mr);
  EXPECT_EQ(resources.get_temporary_mr(), temp_mr);
}

TEST_F(MemoryResourcesTest, SingleArgumentConstructor)
{
  auto output_mr = rmm::mr::get_current_device_resource();

  cudf::memory_resources resources(output_mr);

  // Output MR should be the provided one
  EXPECT_EQ(resources.get_output_mr(), output_mr);

  // Temporary MR should be current device resource (may be same or different)
  // We can't assert equality here since get_current_device_resource_ref()
  // is called at construction time
  auto temp_mr = resources.get_temporary_mr();
  EXPECT_NE(temp_mr, nullptr);
}

TEST_F(MemoryResourcesTest, ImplicitConversion)
{
  auto mr = rmm::mr::get_current_device_resource();

  // This should compile due to implicit conversion from device_async_resource_ref
  auto test_implicit = [](cudf::memory_resources resources) { return resources.get_output_mr(); };

  // Call with device_async_resource_ref - should implicitly convert
  auto result = test_implicit(mr);
  EXPECT_EQ(result, mr);
}

// =============================================================================
// Separate Memory Pool Tests
// =============================================================================

TEST_F(MemoryResourcesTest, SeparateMemoryPools)
{
  // Create two separate pool memory resources
  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::pool_memory_resource output_pool{cuda_mr, 1024 * 1024};  // 1MB initial
  rmm::mr::pool_memory_resource temp_pool{cuda_mr, 1024 * 1024};    // 1MB initial

  cudf::memory_resources resources(&output_pool, &temp_pool);

  // Create a simple column - this should use output_pool for the column data
  auto stream = cudf::get_default_stream();
  auto col    = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED, stream, resources);

  EXPECT_EQ(col->size(), 100);
  EXPECT_EQ(col->type().id(), cudf::type_id::INT32);
}

TEST_F(MemoryResourcesTest, SeparatePoolsWithTracking)
{
  auto cuda_mr = rmm::mr::get_current_device_resource();

  // Create tracking adaptors to monitor allocations
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> temp_tracking{cuda_mr};

  // Reset counters
  output_tracking.reset_allocations();
  temp_tracking.reset_allocations();

  cudf::memory_resources resources(&output_tracking, &temp_tracking);

  // Create a column
  auto stream = cudf::get_default_stream();
  auto col    = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1000, cudf::mask_state::UNALLOCATED, stream, resources);

  // The output column should have allocated from output_mr
  EXPECT_GT(output_tracking.get_bytes_allocated(), 0);

  // Some temporary allocations may have occurred
  // (This is implementation dependent - some operations may not need temp memory)
}

TEST_F(MemoryResourcesTest, StatisticsResourceAdaptor)
{
  auto cuda_mr = rmm::mr::get_current_device_resource();

  // Create statistics adaptors
  rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_memory_resource> output_stats{cuda_mr};
  rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_memory_resource> temp_stats{cuda_mr};

  cudf::memory_resources resources(&output_stats, &temp_stats);

  auto stream = cudf::get_default_stream();

  // Get initial counts
  auto initial_output_allocs = output_stats.get_total_allocated_count();
  auto initial_temp_allocs   = temp_stats.get_total_allocated_count();

  // Create a column
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1000, cudf::mask_state::UNALLOCATED, stream, resources);

  // Output allocations should have increased
  EXPECT_GT(output_stats.get_total_allocated_count(), initial_output_allocs);
}

// =============================================================================
// API Compatibility Tests
// =============================================================================

TEST_F(MemoryResourcesTest, BackwardCompatibilityWithResourceRef)
{
  auto mr = rmm::mr::get_current_device_resource();

  // Old style call - should still work due to implicit conversion
  auto stream = cudf::get_default_stream();
  auto col1   = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED, stream, mr);

  EXPECT_EQ(col1->size(), 100);

  // New style call - explicit memory_resources
  cudf::memory_resources resources(mr);
  auto col2 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED, stream, resources);

  EXPECT_EQ(col2->size(), 100);
}

TEST_F(MemoryResourcesTest, DefaultParameterWorks)
{
  auto stream = cudf::get_default_stream();

  // Call without memory resource parameter - should use default
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED, stream);

  EXPECT_EQ(col->size(), 100);
  EXPECT_EQ(col->type().id(), cudf::type_id::INT32);
}

// =============================================================================
// Functional Tests with Real Operations
// =============================================================================

TEST_F(MemoryResourcesTest, GatherWithSeparateResources)
{
  using namespace cudf::test;

  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::pool_memory_resource output_pool{cuda_mr, 1024 * 1024};
  rmm::mr::pool_memory_resource temp_pool{cuda_mr, 1024 * 1024};

  cudf::memory_resources resources(&output_pool, &temp_pool);

  // Create input data
  fixed_width_column_wrapper<int32_t> col1({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  fixed_width_column_wrapper<int32_t> col2({10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
  cudf::table_view input({col1, col2});

  // Gather map
  fixed_width_column_wrapper<int32_t> gather_map({0, 2, 4, 6, 8});

  // Perform gather
  auto stream = cudf::get_default_stream();
  auto result =
    cudf::gather(input, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, stream, resources);

  // Verify result
  EXPECT_EQ(result->num_columns(), 2);
  EXPECT_EQ(result->num_rows(), 5);

  fixed_width_column_wrapper<int32_t> expected_col1({0, 2, 4, 6, 8});
  fixed_width_column_wrapper<int32_t> expected_col2({10, 12, 14, 16, 18});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected_col1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(1), expected_col2);
}

TEST_F(MemoryResourcesTest, ColumnFactoriesUseSeparateResources)
{
  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> temp_tracking{cuda_mr};

  output_tracking.reset_allocations();
  temp_tracking.reset_allocations();

  cudf::memory_resources resources(&output_tracking, &temp_tracking);

  auto stream = cudf::get_default_stream();

  // Create various types of columns
  auto numeric_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 500, cudf::mask_state::UNALLOCATED, stream, resources);

  auto fixed_point_col =
    cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL32, -2},
                                  500,
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  resources);

  // All output allocations should come from output_tracking
  EXPECT_GT(output_tracking.get_bytes_allocated(), 0);
  EXPECT_GT(output_tracking.get_total_allocated_count(), 0);
}

TEST_F(MemoryResourcesTest, ThreadingResourcesThroughCalls)
{
  using namespace cudf::test;

  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> temp_tracking{cuda_mr};

  cudf::memory_resources resources(&output_tracking, &temp_tracking);

  // Create a column from scalar - this internally calls other column factories
  auto stream = cudf::get_default_stream();
  cudf::numeric_scalar<int32_t> scalar(42, true, stream);

  auto col = cudf::make_column_from_scalar(scalar, 1000, stream, resources);

  EXPECT_EQ(col->size(), 1000);
  EXPECT_EQ(col->type().id(), cudf::type_id::INT32);

  // Verify memory was allocated
  EXPECT_GT(output_tracking.get_bytes_allocated(), 0);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(MemoryResourcesTest, ZeroSizeColumn)
{
  auto mr = rmm::mr::get_current_device_resource();
  cudf::memory_resources resources(mr);

  auto stream = cudf::get_default_stream();
  auto col    = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, resources);

  EXPECT_EQ(col->size(), 0);
  EXPECT_EQ(col->type().id(), cudf::type_id::INT32);
}

TEST_F(MemoryResourcesTest, LargeAllocation)
{
  auto mr = rmm::mr::get_current_device_resource();
  cudf::memory_resources resources(mr);

  auto stream    = cudf::get_default_stream();
  auto large_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                             1000000,  // 1M elements
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             resources);

  EXPECT_EQ(large_col->size(), 1000000);
  EXPECT_EQ(large_col->type().id(), cudf::type_id::INT64);
  EXPECT_EQ(large_col->null_count(), 0);
}

TEST_F(MemoryResourcesTest, NullMaskAllocation)
{
  auto cuda_mr = rmm::mr::get_current_device_resource();

  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> output_tracking{cuda_mr};
  rmm::mr::tracking_resource_adaptor<rmm::mr::cuda_memory_resource> temp_tracking{cuda_mr};

  cudf::memory_resources resources(&output_tracking, &temp_tracking);

  auto stream = cudf::get_default_stream();

  // Create column with null mask
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1000, cudf::mask_state::ALL_NULL, stream, resources);

  EXPECT_EQ(col->size(), 1000);
  EXPECT_EQ(col->null_count(), 1000);
  EXPECT_TRUE(col->nullable());

  // Both data and null mask should be allocated from output resource
  EXPECT_GT(output_tracking.get_bytes_allocated(), 0);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(MemoryResourcesTest, GettersReturnCorrectReferences)
{
  auto mr1 = rmm::mr::get_current_device_resource();
  auto mr2 = rmm::mr::get_current_device_resource();

  cudf::memory_resources resources(mr1, mr2);

  // Getters should return the exact same references
  EXPECT_EQ(resources.get_output_mr(), mr1);
  EXPECT_EQ(resources.get_temporary_mr(), mr2);

  // Should be able to call multiple times
  EXPECT_EQ(resources.get_output_mr(), resources.get_output_mr());
  EXPECT_EQ(resources.get_temporary_mr(), resources.get_temporary_mr());
}

TEST_F(MemoryResourcesTest, GettersAreNoexcept)
{
  auto mr = rmm::mr::get_current_device_resource();
  cudf::memory_resources resources(mr);

  // These should not throw
  EXPECT_NO_THROW(resources.get_output_mr());
  EXPECT_NO_THROW(resources.get_temporary_mr());
}

// =============================================================================
// Copy and Assignment (Implicit) Tests
// =============================================================================

TEST_F(MemoryResourcesTest, CopyConstructor)
{
  auto mr1 = rmm::mr::get_current_device_resource();
  auto mr2 = rmm::mr::get_current_device_resource();

  cudf::memory_resources resources1(mr1, mr2);
  cudf::memory_resources resources2(resources1);

  EXPECT_EQ(resources2.get_output_mr(), mr1);
  EXPECT_EQ(resources2.get_temporary_mr(), mr2);
}

TEST_F(MemoryResourcesTest, CopyAssignment)
{
  auto mr1 = rmm::mr::get_current_device_resource();
  auto mr2 = rmm::mr::get_current_device_resource();
  auto mr3 = rmm::mr::get_current_device_resource();

  cudf::memory_resources resources1(mr1, mr2);
  cudf::memory_resources resources2(mr3, mr3);

  resources2 = resources1;

  EXPECT_EQ(resources2.get_output_mr(), mr1);
  EXPECT_EQ(resources2.get_temporary_mr(), mr2);
}
