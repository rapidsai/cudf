/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

template <typename T>
class GatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GatherTest, cudf::test::NumericTypes);

// This test exercises using different iterator types as gather map inputs
// to cudf::detail::gather -- device_uvector and raw pointers.
TYPED_TEST(GatherTest, GatherDetailDeviceVectorTest)
{
  constexpr cudf::size_type source_size{1000};
  rmm::device_uvector<cudf::size_type> gather_map(source_size, cudf::get_default_stream());
  thrust::sequence(
    rmm::exec_policy_nosync(cudf::get_default_stream()), gather_map.begin(), gather_map.end());

  auto data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);

  cudf::table_view source_table({source_column});

  // test with device vector iterators
  {
    std::unique_ptr<cudf::table> result =
      cudf::detail::gather(source_table,
                           gather_map.begin(),
                           gather_map.end(),
                           cudf::out_of_bounds_policy::DONT_CHECK,
                           cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref());

    for (auto i = 0; i < source_table.num_columns(); ++i) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_table.column(i), result->view().column(i));
    }

    CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
  }

  // test with raw pointers
  {
    std::unique_ptr<cudf::table> result =
      cudf::detail::gather(source_table,
                           gather_map.begin(),
                           gather_map.data() + gather_map.size(),
                           cudf::out_of_bounds_policy::DONT_CHECK,
                           cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref());

    for (auto i = 0; i < source_table.num_columns(); ++i) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_table.column(i), result->view().column(i));
    }

    CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
  }
}

TYPED_TEST(GatherTest, GatherDetailInvalidIndexTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  auto gather_map_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 2) ? -1 : i; });
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(gather_map_data,
                                                             gather_map_data + (source_size * 2));

  cudf::table_view source_table({source_column});
  std::unique_ptr<cudf::table> result =
    cudf::detail::gather(source_table,
                         gather_map,
                         cudf::out_of_bounds_policy::NULLIFY,
                         cudf::detail::negative_index_policy::NOT_ALLOWED,
                         cudf::get_default_stream(),
                         cudf::get_current_device_resource_ref());

  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 2) ? 0 : i; });
  auto expect_valid = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return (i % 2) || (i >= source_size) ? 0 : 1; });
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + (source_size * 2), expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}
