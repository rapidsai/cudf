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
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/detail/copy.hpp>

#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <thrust/equal.h>

#include <gmock/gmock.h>

namespace cudf {
namespace test {

// Property equality
void expect_column_properties_equal(cudf::column_view const& lhs, cudf::column_view const& rhs) {
  EXPECT_EQ(lhs.type(), rhs.type());
  EXPECT_EQ(lhs.size(), rhs.size());
  EXPECT_EQ(lhs.null_count(), rhs.null_count());
  if (lhs.size() > 0) {
     EXPECT_EQ(lhs.nullable(), rhs.nullable());
  }
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

void expect_columns_equal(cudf::column_view const& lhs, cudf::column_view const& rhs,
                          bool print_all_differences) {
  expect_column_properties_equal(lhs, rhs);

  auto d_lhs = cudf::table_device_view::create(table_view{{lhs}});
  auto d_rhs = cudf::table_device_view::create(table_view{{rhs}});

  thrust::device_vector<int> differences(lhs.size());

  auto diff_iter = thrust::copy_if(thrust::device,
                                   thrust::make_counting_iterator(0),
                                   thrust::make_counting_iterator(lhs.size()),
                                   differences.begin(),
                                   corresponding_rows_unequal(*d_lhs, *d_rhs));

  CUDA_TRY(cudaDeviceSynchronize());

  differences.resize(thrust::distance(differences.begin(), diff_iter));

  if (diff_iter > differences.begin()) {
    if (print_all_differences) {
      //
      //  If there are differences, display them all
      //
      std::ostringstream buffer;
      buffer << "differences:" << std::endl;
      
      cudf::table_view source_table ({lhs, rhs});

      fixed_width_column_wrapper<int32_t> diff_column(differences.begin(), differences.end());

      std::unique_ptr<cudf::experimental::table> diff_table = cudf::experimental::gather(source_table,
											 diff_column);
      
      //
      //  Need to pull back the differences
      //
      std::vector<std::string> h_left_strings = to_strings(diff_table->get_column(0));
      std::vector<std::string> h_right_strings = to_strings(diff_table->get_column(1));

      for (size_t i = 0 ; i < differences.size() ; ++i) {
          buffer << "lhs[" << differences[i] << "] = " << h_left_strings[i]
                 << ", rhs[" << differences[i] << "] = " << h_right_strings[i] << std::endl;
      }

      EXPECT_EQ(differences.size(), size_t{0}) << buffer.str();
    } else {
      //
      //  If there are differences, just display the first one
      //
      int index = differences[0];

      auto diff_lhs = cudf::experimental::detail::slice(lhs, index, index+1);
      auto diff_rhs = cudf::experimental::detail::slice(rhs, index, index+1);

      std::vector<std::string> h_left_strings = to_strings(diff_lhs);
      std::vector<std::string> h_right_strings = to_strings(diff_rhs);

      EXPECT_EQ(differences.size(), size_t{0}) << "first difference: "
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

// copy column bitmask to host (used by to_host())
std::vector<bitmask_type> bitmask_to_host(cudf::column_view const& c) {
  if (c.nullable()) {
    auto num_bitmasks = bitmask_allocation_size_bytes(c.size()) / sizeof(bitmask_type);
    std::vector<bitmask_type> host_bitmask(num_bitmasks);
    if (c.offset()==0) {
        CUDA_TRY(cudaMemcpy(host_bitmask.data(), c.null_mask(), num_bitmasks * sizeof(bitmask_type),
                            cudaMemcpyDeviceToHost));
    } else {
        auto mask = copy_bitmask(c.null_mask(), c.offset(), c.offset()+c.size());
        CUDA_TRY(cudaMemcpy(host_bitmask.data(), mask.data(), num_bitmasks * sizeof(bitmask_type),
                            cudaMemcpyDeviceToHost));
    }

    return host_bitmask;
  }
  else {
    return std::vector<bitmask_type>{};
  }
}


struct column_view_printer {
  template <typename Element, typename std::enable_if_t<is_numeric<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string> & out) {
    auto h_data = cudf::test::to_host<Element>(col);

    out.resize(col.size());

    if (col.nullable()) {
      std::transform(thrust::make_counting_iterator(size_type{0}),
                     thrust::make_counting_iterator(col.size()),
                     out.begin(),
                     [&h_data](auto idx) {
                       return bit_is_set(h_data.second.data(), idx) ? std::to_string(h_data.first[idx]) : std::string("NULL");
                     });
    } else {
      std::transform(h_data.first.begin(), h_data.first.end(), out.begin(), [](Element el) {
          return std::to_string(el);
        });
    }
  }

  template <typename Element, typename std::enable_if_t<is_timestamp<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string> & out) {
    //
    //  For timestamps, convert timestamp column to column of strings, then
    //  call string version
    //
    auto col_as_strings = cudf::strings::from_timestamps(col);

    this->template operator()<cudf::string_view>(*col_as_strings, out);
  }

  template <typename Element, typename std::enable_if_t<std::is_same<Element, cudf::string_view>::value>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string> & out) {
    //
    //  Implementation for strings, call special to_host variant
    //
    auto h_data = cudf::test::to_host<std::string>(col);

    out.resize(col.size());
    if (col.nullable()) {
      std::transform(thrust::make_counting_iterator(size_type{0}),
                     thrust::make_counting_iterator(col.size()),
                     out.begin(),
                     [&h_data](auto idx) {
                       return bit_is_set(h_data.second.data(), idx) ? h_data.first[idx] : std::string("NULL");
                     });
    } else {
      out = std::move(h_data.first);
    }
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

std::string to_string(cudf::column_view const& col, std::string const& delimiter) {

  std::ostringstream buffer;
  std::vector<std::string> h_data = to_strings(col);

  std::copy(h_data.begin(), h_data.end() - 1, std::ostream_iterator<std::string>(buffer, delimiter.c_str()));
  buffer << h_data.back();

  return buffer.str();
}

void print(cudf::column_view const& col, std::ostream &os, std::string const& delimiter) {
  os << to_string(col, delimiter);
}

}  // namespace test
}  // namespace cudf
