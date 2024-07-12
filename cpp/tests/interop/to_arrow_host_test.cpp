/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nanoarrow_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

struct ToArrowHostDeviceTest : public cudf::test::BaseFixture {
  void compare_arrays(ArrowArrayView const* expected, ArrowArrayView const* actual)
  {
    EXPECT_EQ(expected->length, actual->length);
    EXPECT_EQ(expected->null_count, actual->null_count);
    EXPECT_EQ(expected->offset, actual->offset);
    EXPECT_EQ(expected->n_children, actual->n_children);
    EXPECT_EQ(expected->array->n_buffers, actual->array->n_buffers);

    for (int64_t i = 0; i < expected->array->n_buffers; ++i) {
      auto expected_buf = expected->buffer_views[i];
      auto actual_buf   = actual->buffer_views[i];

      EXPECT_TRUE(
        0 == std::memcmp(expected_buf.data.data, actual_buf.data.data, expected_buf.size_bytes));
    }

    if (expected->dictionary != nullptr) {
      EXPECT_NE(nullptr, actual->dictionary);
      SCOPED_TRACE("dictionary");
      compare_arrays(expected->dictionary, actual->dictionary);
    } else {
      EXPECT_EQ(nullptr, actual->dictionary);
    }

    if (expected->n_children == 0) {
      EXPECT_EQ(nullptr, actual->children);
    } else {
      for (int64_t i = 0; i < expected->n_children; ++i) {
        SCOPED_TRACE("child " + std::to_string(i));
        compare_arrays(expected->children[i], actual->children[i]);
      }
    }
  }
};

// template <typename T>
// struct ToArrowHostDeviceTestDurationsTest : public cudf::test::BaseFixture {};

// TYPED_TEST_SUITE(ToArrowHostDeviceTestDurationsTest, cudf::test::DurationTypes);

TEST_F(ToArrowHostDeviceTest, EmptyTable)
{
  auto [tbl, schema, arr] = get_nanoarrow_host_tables(0);

  auto got_arrow_host = cudf::to_arrow_host(tbl->view());
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, arr.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);
}
