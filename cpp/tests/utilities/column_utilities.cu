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
  EXPECT_EQ(lhs.nullable(), rhs.nullable());
  EXPECT_EQ(lhs.has_nulls(), rhs.has_nulls());
  EXPECT_EQ(lhs.num_children(), rhs.num_children());
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

void expect_columns_equal(cudf::column_view lhs, cudf::column_view rhs, bool all) {
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

  if (diff_iter > differences.begin()) {
    int difference_count = diff_iter - differences.begin();

    if (all) {
      //
      //  If there are differences, display them all
      //
      std::ostringstream buffer;
      buffer << "differences:" << std::endl;
      
#ifdef USE_GATHER
      cudf::table_view source_table ({lhs, rhs});

      std::unique_ptr<cudf::experimental::table> diff_table = cudf::experimental::gather(source_table,
											 differences);

      //
      //  Need to pull back the differences
      //
      std::vector<std::string> h_left_strings = to_strings(diff_table->column(0));
      std::vector<std::string> h_right_strings = to_strings(diff_table->column(1));

      for (int i = 0 ; i < differences_count ; ++i) {
          buffer << "lhs[" << differences[i] << "] = " << h_left_strings[i]
                 << ", rhs[" << differences[i] << "] = " << h_right_strings[i] << std::endl;
      }
#else
      std::vector<std::string> h_left_strings = to_strings(lhs);
      std::vector<std::string> h_right_strings = to_strings(rhs);

      std::for_each(differences.begin(), diff_iter, [&buffer, &h_left_strings, &h_right_strings](int idx) {
          buffer << "lhs[" << idx << "] = " << h_left_strings[idx]
                 << ", rhs[" << idx << "] = " << h_right_strings[idx] << std::endl;
        });
#endif

      EXPECT_EQ(difference_count, 0) << buffer.str();
    } else {
      //
      //  If there are differences, just display the first one
      //
      int index = differences[0];

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

      EXPECT_EQ(difference_count, 0) << "first difference: "
                                     << "lhs[" << index << "] = "
                                     << to_string(diff_lhs, "")
                                     << ", rhs[" << index << "] = "
                                     << to_string(diff_rhs, "");
    }
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
  void operator()(cudf::column_view const& col, std::vector<std::string> & out) {
    auto h_data = cudf::test::to_host<Element>(col);

    out.resize(col.size());

    if (col.nullable()) {
      size_type index = 0;
      std::transform(h_data.first.begin(), h_data.first.end(), out.begin(), [&h_data, &index](Element el) {
          return (bit_is_set(h_data.second.data(), index++)) ? std::to_string(el) : std::string("@");
        });
    } else {
      std::transform(h_data.first.begin(), h_data.first.end(), out.begin(), [](Element el) {
          return std::to_string(el);
        });
    }
  }

  template <typename Element, typename std::enable_if_t<not is_numeric<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string> & out) {
    CUDF_FAIL("printing not currently enabled for non-numeric arguments");
  }
};

std::vector<std::string> to_strings(cudf::column_view const& col) {
  std::vector<std::string> reply;

  cudf::experimental::type_dispatcher(col.type(),
                                      column_view_printer{}, 
                                      col,
                                      reply);

  return reply;
}

std::string to_string(cudf::column_view const& col, const char *delimiter) {

  std::ostringstream buffer;
  std::vector<std::string> h_data = to_strings(col);

  std::copy(h_data.begin(), h_data.end() - 1, std::ostream_iterator<std::string>(buffer, delimiter));
  buffer << h_data.back();

  return buffer.str();
}

void print(std::ostream &os, cudf::column_view const& col, const char *delimiter) {
  os << to_string(col, delimiter);
}

}  // namespace test
}  // namespace cudf
