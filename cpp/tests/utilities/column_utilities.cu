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
#include <thrust/logical.h>

#include <gmock/gmock.h>

namespace cudf {
namespace test {

// Property comparison
template <bool check_exact_equality>
void column_property_comparison(cudf::column_view const& lhs, cudf::column_view const& rhs) {
  EXPECT_EQ(lhs.type(), rhs.type());
  EXPECT_EQ(lhs.size(), rhs.size());
  EXPECT_EQ(lhs.null_count(), rhs.null_count());
  if (lhs.size() > 0 and check_exact_equality) {
    EXPECT_EQ(lhs.nullable(), rhs.nullable());
  }
  EXPECT_EQ(lhs.num_children(), rhs.num_children());
}

void expect_column_properties_equal(column_view const& lhs, column_view const& rhs) {
  column_property_comparison<true>(lhs, rhs);
}

void expect_column_properties_equivalent(column_view const& lhs, column_view const& rhs) {
  column_property_comparison<false>(lhs, rhs);
}

class corresponding_rows_unequal {
  table_device_view d_lhs; 
  table_device_view d_rhs;
public:
  corresponding_rows_unequal(table_device_view d_lhs, table_device_view d_rhs)
  : d_lhs(d_lhs),
    d_rhs(d_rhs),
    comp(d_lhs, d_rhs)
  {
    CUDF_EXPECTS(d_lhs.num_columns() == 1 and d_rhs.num_columns() == 1,
                 "Unsupported number of columns");
  }
  
  struct typed_element_unequal {
    template <typename T>
    __device__ std::enable_if_t<std::is_floating_point<T>::value, bool>
    operator()(column_device_view const& lhs,
               column_device_view const& rhs,
               size_type index)
    {
      if (lhs.is_valid(index) and rhs.is_valid(index)) {
        int ulp = 4; // value taken from google test
        T x = lhs.element<T>(index);
        T y = rhs.element<T>(index);
        return std::abs(x-y) > std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
            && std::abs(x-y) >= std::numeric_limits<T>::min();
      } else {
        // if either is null, then the inequality was checked already
        return true;
      }
    }

    template <typename T, typename... Args>
    __device__ std::enable_if_t<not std::is_floating_point<T>::value, bool>
    operator()(Args... args) {
      // Non-floating point inequality is checked already
      return true;
  }
  };
  
  cudf::experimental::row_equality_comparator<true> comp;
    
  __device__ bool operator()(size_type index) {
    if (not comp(index, index)) {
      return thrust::any_of(thrust::seq,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(d_lhs.num_columns()),
        [this, index] (auto i) {
          auto lhs_col = this->d_lhs.column(i);
          auto rhs_col = this->d_rhs.column(i);
          return experimental::type_dispatcher(lhs_col.type(), 
                                               typed_element_unequal{},
                                               lhs_col, rhs_col, index);
        });
    }
    return false;
  }
};

class corresponding_rows_not_equivalent {
  table_device_view d_lhs; 
  table_device_view d_rhs;
public:
  corresponding_rows_not_equivalent(table_device_view d_lhs, table_device_view d_rhs)
  : d_lhs(d_lhs),
    d_rhs(d_rhs),
    comp(d_lhs, d_rhs)
  {
    CUDF_EXPECTS(d_lhs.num_columns() == 1 and d_rhs.num_columns() == 1,
                 "Unsupported number of columns");
  }
  
  struct typed_element_not_equivalent {
    template <typename T>
    __device__ std::enable_if_t<std::is_floating_point<T>::value, bool>
    operator()(column_device_view const& lhs,
               column_device_view const& rhs,
               size_type index)
    {
      if (lhs.is_valid(index) and rhs.is_valid(index)) {
        int ulp = 4; // value taken from google test
        T x = lhs.element<T>(index);
        T y = rhs.element<T>(index);
        return std::abs(x-y) > std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
            && std::abs(x-y) >= std::numeric_limits<T>::min();
      } else {
        // if either is null, then the inequality was checked already
        return true;
      }
    }

    template <typename T, typename... Args>
    __device__ std::enable_if_t<not std::is_floating_point<T>::value, bool>
    operator()(Args... args) {
      // Non-floating point inequality is checked already
      return true;
    }
  };
  
  cudf::experimental::row_equality_comparator<true> comp;
    
  __device__ bool operator()(size_type index) {
    if (not comp(index, index)) {
      auto lhs_col = this->d_lhs.column(0);
      auto rhs_col = this->d_rhs.column(0);
      return experimental::type_dispatcher(lhs_col.type(), 
                                           typed_element_not_equivalent{},
                                           lhs_col, rhs_col, index);
    }
    return false;
  }
};

namespace {

template <bool check_exact_equality>
void column_comparison(cudf::column_view const& lhs, cudf::column_view const& rhs,
                       bool print_all_differences) {
  column_property_comparison<check_exact_equality>(lhs, rhs);

  using ComparatorType = std::conditional_t<check_exact_equality, 
                                            corresponding_rows_unequal,
                                            corresponding_rows_not_equivalent>;

  auto d_lhs = cudf::table_device_view::create(table_view{{lhs}});
  auto d_rhs = cudf::table_device_view::create(table_view{{rhs}});

  // TODO (dm): handle floating point equality
  thrust::device_vector<int> differences(lhs.size());

  auto diff_iter = thrust::copy_if(thrust::device,
                                   thrust::make_counting_iterator(0),
                                   thrust::make_counting_iterator(lhs.size()),
                                   differences.begin(),
                                   ComparatorType(*d_lhs, *d_rhs));

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

} // namespace anonymous

void expect_columns_equal(cudf::column_view const& lhs, cudf::column_view const& rhs,
                          bool print_all_differences) 
{
  column_comparison<true>(lhs, rhs, print_all_differences);
}

void expect_columns_equivalent(cudf::column_view const& lhs,
                               cudf::column_view const& rhs,
                               bool print_all_differences) 
{
  column_comparison<false>(lhs, rhs, print_all_differences);
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
