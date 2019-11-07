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

#include "column_utilities.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/bit.hpp>

#include <tests/utilities/cudf_gtest.hpp>

#include <thrust/equal.h>

#include <gmock/gmock.h>

namespace cudf {
namespace test {

// Property equality
void expect_column_properties_equal(cudf::column_view lhs, cudf::column_view rhs) {
  EXPECT_EQ(lhs.type(), rhs.type());
  EXPECT_EQ(lhs.size(), rhs.size());
  EXPECT_EQ(lhs.null_count(), rhs.null_count());
  if (lhs.size() > 0) {
      EXPECT_EQ(lhs.nullable(), rhs.nullable());
  }
  EXPECT_EQ(lhs.has_nulls(), rhs.has_nulls());
  EXPECT_EQ(lhs.num_children(), rhs.num_children());
}

// Verify elementwise equality
bool column_values_equal(cudf::column_view lhs, cudf::column_view rhs) {
  auto d_lhs = cudf::table_device_view::create(table_view{{lhs}});
  auto d_rhs = cudf::table_device_view::create(table_view{{rhs}});

  bool reply = thrust::equal(thrust::device,
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(lhs.size()),
                             thrust::make_counting_iterator(0),
                             cudf::experimental::row_equality_comparator<true>{*d_lhs, *d_rhs});

  CUDA_TRY(cudaDeviceSynchronize());

  return reply;
}

class corresponding_rows_unequal {
public:
  corresponding_rows_unequal(table_device_view d_lhs, table_device_view d_rhs): comp(d_lhs, d_rhs) {
  }
  
  cudf::experimental::row_equality_comparator<true> comp;
    
  __device__ bool operator()(size_type index) {
    return !comp(index, index);
  }
};

void expect_columns_equal(cudf::column_view lhs, cudf::column_view rhs) {
  expect_column_properties_equal(lhs, rhs);

  auto d_lhs = cudf::table_device_view::create(table_view{{lhs}});
  auto d_rhs = cudf::table_device_view::create(table_view{{rhs}});

  thrust::device_vector<int> differences;
  differences.reserve(lhs.size());

  auto diff_iter = thrust::copy_if(thrust::device,
                                   thrust::make_counting_iterator(0),
                                   thrust::make_counting_iterator(lhs.size()),
                                   differences.begin(),
                                   corresponding_rows_unequal(*d_lhs, *d_rhs));

  CUDA_TRY(cudaDeviceSynchronize());

  //
  //  If there are differences, let's display the first one
  //
  if (diff_iter > differences.begin()) {
    int index = differences[0];
    int count = diff_iter - differences.begin();

    cudf::column_view diff_lhs(lhs.type(),
                               1,
                               lhs.data<void *>(),
                               lhs.null_mask(),
                               0,
                               index);
                               
    cudf::column_view diff_rhs(rhs.type(),
                               1,
                               rhs.data<void *>(),
                               rhs.null_mask(),
                               0,
                               index);

    EXPECT_PRED_FORMAT1(([diff_lhs, diff_rhs, count]
                         (const char *m_expr, int m) {
                           return ::testing::AssertionFailure()
                             << "expect_columns_equal failed with ("
                             << count
                             << ") differences: "
                             << "lhs[" << m << "] = "
                             << column_view_to_str(diff_lhs, "")
                             << ", rhs[" << m << "] = "
                             << column_view_to_str(diff_rhs, "");
                         }), index);
  
  }
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

struct column_view_printer {
  template <typename Element, typename std::enable_if_t<is_numeric<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, const char *delimiter, std::ostream &ostream) {

    cudf::size_type num_rows = col.size();
    std::vector<Element> h_data(num_rows);
    CUDA_TRY(cudaMemcpy(h_data.data(), col.data<void *>(),
                        num_rows * sizeof(Element),
                        cudaMemcpyDeviceToHost));

    if (col.nullable()) {
      cudf::size_type null_size = (num_rows - 1 + cudf::detail::size_in_bits<bitmask_type>()) /
        cudf::detail::size_in_bits<bitmask_type>();

      std::vector<bitmask_type> h_null(null_size);
      CUDA_TRY(cudaMemcpy(h_null.data(), col.null_mask(),
                          cudf::detail::size_in_bits<bitmask_type>() * null_size,
                          cudaMemcpyDeviceToHost));

      thrust::for_each(thrust::host,
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(num_rows),
                       [&h_data, &h_null,
                        &delimiter, &ostream](int index) {
                         if (bit_is_set(h_null.data(), index)) {
                           ostream << h_data[index];
                         } else {
                           ostream << "@";
                         }
                       });
    } else {
      thrust::copy(h_data.begin(),
                   h_data.end(),
                   std::ostream_iterator<Element>(ostream, delimiter));
    }
  }

  template <typename Element, typename std::enable_if_t<not is_numeric<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, const char *delimiter, std::ostream &ostream) {
    CUDF_FAIL("printing not currently enabled for non-numeric arguments");
  }
};

std::string column_view_to_str(cudf::column_view const& col, const char *delimiter) {

  std::ostringstream buffer;

  cudf::experimental::type_dispatcher(col.type(),
                                      column_view_printer{}, 
                                      col,
                                      delimiter,
                                      buffer);

  return buffer.str();
}

}  // namespace test
}  // namespace cudf
