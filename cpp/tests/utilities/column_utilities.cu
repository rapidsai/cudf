/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "column_utilities.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <tests/utilities/cudf_gtest.hpp>

#include <thrust/equal.h>

#include <gmock/gmock.h>

namespace cudf {
namespace test {

// Verify elementwise equality
void expect_columns_equal(cudf::column_view lhs, cudf::column_view rhs) {
  EXPECT_EQ(lhs.type(), rhs.type());
  EXPECT_EQ(lhs.size(), rhs.size());
  EXPECT_EQ(lhs.null_count(), rhs.null_count());
  EXPECT_EQ(lhs.nullable(), rhs.nullable());
  EXPECT_EQ(lhs.has_nulls(), rhs.has_nulls());
  EXPECT_EQ(lhs.num_children(), rhs.num_children());

  auto d_lhs = cudf::table_device_view::create(table_view{{lhs}});
  auto d_rhs = cudf::table_device_view::create(table_view{{rhs}});

  EXPECT_TRUE(
      thrust::equal(thrust::device, thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(lhs.size()),
                    thrust::make_counting_iterator(0),
                    cudf::exp::row_equality_comparator<true>{*d_lhs, *d_rhs}));

  CUDA_TRY(cudaDeviceSynchronize());
}

// Bitwise equality
void expect_equal_buffers(void const* lhs, void const* rhs,
                          std::size_t size_bytes) {
  if (size_bytes > 0) {
    EXPECT_NE(nullptr, lhs);
    EXPECT_NE(nullptr, rhs);
  }
  auto typed_lhs = static_cast<char const*>(lhs);
  auto typed_rhs = static_cast<char const*>(rhs);
  EXPECT_TRUE(thrust::equal(thrust::device, typed_lhs, typed_lhs + size_bytes,
                            typed_rhs));
}

}  // namespace test
}  // namespace cudf