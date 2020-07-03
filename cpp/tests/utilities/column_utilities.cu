/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include "detail/column_utilities.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/bit.hpp>

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>

#include <jit/type.h>

#include <thrust/equal.h>
#include <thrust/logical.h>

#include <numeric>

namespace cudf {
namespace test {

namespace {

template <bool check_exact_equality>
struct column_property_comparator {
  template <typename T>
  void operator()(cudf::column_view const& lhs, cudf::column_view const& rhs)
  {
    EXPECT_EQ(lhs.type(), rhs.type());
    EXPECT_EQ(lhs.size(), rhs.size());
    if (lhs.size() > 0 && check_exact_equality) { EXPECT_EQ(lhs.nullable(), rhs.nullable()); }
    EXPECT_EQ(lhs.num_children(), rhs.num_children());

    // only recurse for true nested types.
    // - strings are an odd case of not being a nested type which do have children. but because
    //   of the way strings handle offsets (sliced/split columns), direct comparison between two
    //   sets of child columns can produce false failures - the sizes may not match.  the truly
    //   correct way to do this would be to implement a specialization for strings (and
    //   dictionaries, lists, etc) that explicitly understand this structure.  but for now, this
    //   seems to be ok.
    if (cudf::is_nested<T>()) {
      for (size_type idx = 0; idx < lhs.num_children(); idx++) {
        cudf::type_dispatcher(lhs.child(idx).type(),
                              column_property_comparator<check_exact_equality>{},
                              lhs.child(idx),
                              rhs.child(idx));
      }
    }
  }
};

class corresponding_rows_unequal {
 public:
  corresponding_rows_unequal(table_device_view d_lhs, table_device_view d_rhs) : comp(d_lhs, d_rhs)
  {
  }

  cudf::row_equality_comparator<true> comp;

  __device__ bool operator()(size_type index) { return !comp(index, index); }
};

class corresponding_rows_not_equivalent {
  table_device_view d_lhs;
  table_device_view d_rhs;

 public:
  corresponding_rows_not_equivalent(table_device_view d_lhs, table_device_view d_rhs)
    : d_lhs(d_lhs), d_rhs(d_rhs), comp(d_lhs, d_rhs)
  {
    CUDF_EXPECTS(d_lhs.num_columns() == 1 and d_rhs.num_columns() == 1,
                 "Unsupported number of columns");
  }

  struct typed_element_not_equivalent {
    template <typename T>
    __device__ std::enable_if_t<std::is_floating_point<T>::value, bool> operator()(
      column_device_view const& lhs, column_device_view const& rhs, size_type index)
    {
      if (lhs.is_valid(index) and rhs.is_valid(index)) {
        int ulp = 4;  // value taken from google test
        T x     = lhs.element<T>(index);
        T y     = rhs.element<T>(index);
        return std::abs(x - y) > std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp &&
               std::abs(x - y) >= std::numeric_limits<T>::min();
      } else {
        // if either is null, then the inequality was checked already
        return true;
      }
    }

    template <typename T, typename... Args>
    __device__ std::enable_if_t<not std::is_floating_point<T>::value, bool> operator()(Args... args)
    {
      // Non-floating point inequality is checked already
      return true;
    }
  };

  cudf::row_equality_comparator<true> comp;

  __device__ bool operator()(size_type index)
  {
    if (not comp(index, index)) {
      auto lhs_col = this->d_lhs.column(0);
      auto rhs_col = this->d_rhs.column(0);
      return type_dispatcher(
        lhs_col.type(), typed_element_not_equivalent{}, lhs_col, rhs_col, index);
    }
    return false;
  }
};

void print_differences(thrust::device_vector<int> const& differences,
                       column_view const& lhs,
                       column_view const& rhs,
                       bool print_all_differences,
                       int depth)
{
  if (differences.size() <= 0) { return; }

  std::string depth_str = depth > 0 ? "depth " + std::to_string(depth) + std::string("\n") : "";

  if (print_all_differences) {
    //  If there are differences, display them all
    std::ostringstream buffer;
    buffer << depth_str << "differences:" << std::endl;

    cudf::table_view source_table({lhs, rhs});

    fixed_width_column_wrapper<int32_t> diff_column(differences.begin(), differences.end());

    std::unique_ptr<cudf::table> diff_table = cudf::gather(source_table, diff_column);

    //  Need to pull back the differences
    std::vector<std::string> h_left_strings  = to_strings(diff_table->get_column(0));
    std::vector<std::string> h_right_strings = to_strings(diff_table->get_column(1));

    for (size_t i = 0; i < differences.size(); ++i) {
      buffer << depth_str << "lhs[" << differences[i] << "] = " << h_left_strings[i] << ", rhs["
             << differences[i] << "] = " << h_right_strings[i] << std::endl;
    }

    EXPECT_EQ(differences.size(), size_t{0}) << buffer.str();
  } else {
    //  If there are differences, just display the first one
    int index = differences[0];

    auto diff_lhs = cudf::detail::slice(lhs, index, index + 1);
    auto diff_rhs = cudf::detail::slice(rhs, index, index + 1);

    std::vector<std::string> h_left_strings  = to_strings(diff_lhs);
    std::vector<std::string> h_right_strings = to_strings(diff_rhs);

    EXPECT_EQ(differences.size(), size_t{0})
      << depth_str << "first difference: "
      << "lhs[" << index << "] = " << to_string(diff_lhs, "") << ", rhs[" << index
      << "] = " << to_string(diff_rhs, "");
  }
}

// non-nested column types
template <typename T, bool check_exact_equality>
struct column_comparator_impl {
  void operator()(column_view const& lhs,
                  column_view const& rhs,
                  bool print_all_differences,
                  int depth)
  {
    using ComparatorType = std::conditional_t<check_exact_equality,
                                              corresponding_rows_unequal,
                                              corresponding_rows_not_equivalent>;

    auto d_lhs = cudf::table_device_view::create(table_view{{lhs}});
    auto d_rhs = cudf::table_device_view::create(table_view{{rhs}});

    // worst case - everything is different
    thrust::device_vector<int> differences(lhs.size());

    auto diff_iter = thrust::copy_if(thrust::device,
                                     thrust::make_counting_iterator(0),
                                     thrust::make_counting_iterator(lhs.size()),
                                     differences.begin(),
                                     ComparatorType(*d_lhs, *d_rhs));

    // shrink back down
    differences.resize(thrust::distance(differences.begin(), diff_iter));
    print_differences(differences, lhs, rhs, print_all_differences, depth);
  }
};

// forward declaration for nested-type recursion.
template <bool check_exact_equality>
struct column_comparator;

// specialization for list columns
template <bool check_exact_equality>
struct column_comparator_impl<list_view, check_exact_equality> {
  void operator()(column_view const& lhs,
                  column_view const& rhs,
                  bool print_all_differences,
                  int depth)
  {
    lists_column_view lhs_l(lhs);
    lists_column_view rhs_l(rhs);

    // using the row_equality_operator directly on a list column is a bad idea for several
    // reasons:
    // - at the moment, the row_equality_operator doesn't support lists
    //
    // - if it -did-, a "row" in a list column can itself be nested.  so to do a row
    //   comparison involves actually recursing through the hierarchy of data. this recursion
    //   would be happening for each row compared, which is algorithmically terrible.
    //
    // Instead, we can simply walk the hierarchy once, checking each pair of offset columns for
    // equivalency and then finally checking the leaves, which are not nested types.
    cudf::type_dispatcher(lhs_l.offsets().type(),
                          column_comparator<check_exact_equality>{},
                          lhs_l.offsets(),
                          rhs_l.offsets(),
                          print_all_differences,
                          depth);
    cudf::type_dispatcher(lhs_l.child().type(),
                          column_comparator<check_exact_equality>{},
                          lhs_l.child(),
                          rhs_l.child(),
                          print_all_differences,
                          depth + 1);

    // TODO:  to display differences between list columns what we really want to do is
    //        - if there are differences in the leaf values, display those.
    //
    //        otherwise
    //
    //        - determine the first level at which there are list differences (via the offsets),
    //          do a gather on those rows and display them.
  }
};

template <bool check_exact_equality>
struct column_comparator {
  template <typename T>
  void operator()(column_view const& lhs,
                  column_view const& rhs,
                  bool print_all_differences,
                  int depth = 0)
  {
    // compare properties
    cudf::type_dispatcher(lhs.type(), column_property_comparator<check_exact_equality>{}, lhs, rhs);

    // compare values
    column_comparator_impl<T, check_exact_equality> comparator{};
    comparator(lhs, rhs, print_all_differences, depth);
  }
};

}  // namespace

/**
 * @copydoc cudf::test::expect_column_properties_equal
 *
 */
void expect_column_properties_equal(column_view const& lhs, column_view const& rhs)
{
  cudf::type_dispatcher(lhs.type(), column_property_comparator<true>{}, lhs, rhs);
}

/**
 * @copydoc cudf::test::expect_column_properties_equivalent
 *
 */
void expect_column_properties_equivalent(column_view const& lhs, column_view const& rhs)
{
  cudf::type_dispatcher(lhs.type(), column_property_comparator<false>{}, lhs, rhs);
}

/**
 * @copydoc cudf::test::expect_columns_equal
 *
 */
void expect_columns_equal(cudf::column_view const& lhs,
                          cudf::column_view const& rhs,
                          bool print_all_differences)
{
  cudf::type_dispatcher(lhs.type(), column_comparator<true>{}, lhs, rhs, print_all_differences);
}

/**
 * @copydoc cudf::test::expect_columns_equivalent
 *
 */
void expect_columns_equivalent(cudf::column_view const& lhs,
                               cudf::column_view const& rhs,
                               bool print_all_differences)
{
  cudf::type_dispatcher(lhs.type(), column_comparator<false>{}, lhs, rhs, print_all_differences);
}

/**
 * @copydoc cudf::test::expect_equal_buffers
 *
 */
void expect_equal_buffers(void const* lhs, void const* rhs, std::size_t size_bytes)
{
  if (size_bytes > 0) {
    EXPECT_NE(nullptr, lhs);
    EXPECT_NE(nullptr, rhs);
  }
  auto typed_lhs = static_cast<char const*>(lhs);
  auto typed_rhs = static_cast<char const*>(rhs);
  EXPECT_TRUE(thrust::equal(thrust::device, typed_lhs, typed_lhs + size_bytes, typed_rhs));
}

/**
 * @copydoc cudf::test::bitmask_to_host
 *
 */
std::vector<bitmask_type> bitmask_to_host(cudf::column_view const& c)
{
  if (c.nullable()) {
    auto num_bitmasks = bitmask_allocation_size_bytes(c.size()) / sizeof(bitmask_type);
    std::vector<bitmask_type> host_bitmask(num_bitmasks);
    if (c.offset() == 0) {
      CUDA_TRY(cudaMemcpy(host_bitmask.data(),
                          c.null_mask(),
                          num_bitmasks * sizeof(bitmask_type),
                          cudaMemcpyDeviceToHost));
    } else {
      auto mask = copy_bitmask(c.null_mask(), c.offset(), c.offset() + c.size());
      CUDA_TRY(cudaMemcpy(host_bitmask.data(),
                          mask.data(),
                          num_bitmasks * sizeof(bitmask_type),
                          cudaMemcpyDeviceToHost));
    }

    return host_bitmask;
  } else {
    return std::vector<bitmask_type>{};
  }
}

namespace {

template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
static auto numeric_to_string_precise(T value)
{
  return std::to_string(value);
}

template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static auto numeric_to_string_precise(T value)
{
  std::ostringstream o;
  o << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
  return o.str();
}

std::string get_nested_type_str(cudf::column_view const& view)
{
  if (view.type().id() == cudf::type_id::LIST) {
    lists_column_view lcv(view);
    return cudf::jit::get_type_name(view.type()) + "<" +
           (lcv.size() > 0 ? get_nested_type_str(lcv.child()) : "") + ">";
  }
  return cudf::jit::get_type_name(view.type());
}

struct column_view_printer {
  template <typename Element, typename std::enable_if_t<is_numeric<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    auto h_data = cudf::test::to_host<Element>(col);

    out.resize(col.size());

    if (col.nullable()) {
      std::transform(thrust::make_counting_iterator(size_type{0}),
                     thrust::make_counting_iterator(col.size()),
                     out.begin(),
                     [&h_data](auto idx) {
                       return bit_is_set(h_data.second.data(), idx)
                                ? numeric_to_string_precise(h_data.first[idx])
                                : std::string("NULL");
                     });

    } else {
      std::transform(h_data.first.begin(), h_data.first.end(), out.begin(), [](Element el) {
        return numeric_to_string_precise(el);
      });
    }
  }

  template <typename Element, typename std::enable_if_t<is_timestamp<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    //
    //  For timestamps, convert timestamp column to column of strings, then
    //  call string version
    //
    auto col_as_strings = cudf::strings::from_timestamps(col);
    if (col_as_strings->size() == 0) { return; }

    this->template operator()<cudf::string_view>(*col_as_strings, out, indent);
  }

  template <typename Element,
            typename std::enable_if_t<std::is_same<Element, cudf::string_view>::value>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    //
    //  Implementation for strings, call special to_host variant
    //
    auto h_data = cudf::test::to_host<std::string>(col);

    out.resize(col.size());
    std::transform(thrust::make_counting_iterator(size_type{0}),
                   thrust::make_counting_iterator(col.size()),
                   out.begin(),
                   [&h_data](auto idx) {
                     return h_data.second.empty() || bit_is_set(h_data.second.data(), idx)
                              ? h_data.first[idx]
                              : std::string("NULL");
                   });
  }

  template <typename Element,
            typename std::enable_if_t<std::is_same<Element, cudf::dictionary32>::value>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    cudf::dictionary_column_view dictionary(col);
    if (col.size() == 0) return;
    std::vector<std::string> keys    = to_strings(dictionary.keys());
    std::vector<std::string> indices = to_strings({cudf::data_type{cudf::type_id::INT32},
                                                   dictionary.size(),
                                                   dictionary.indices().head<int32_t>(),
                                                   dictionary.null_mask(),
                                                   dictionary.null_count(),
                                                   dictionary.offset()});
    out.insert(out.end(), keys.begin(), keys.end());
    if (!indices.empty()) {
      std::string first = "\x08 : " + indices.front();  // use : as delimiter
      out.push_back(first);                             // between keys and indices
      out.insert(out.end(), indices.begin() + 1, indices.end());
    }
  }

  template <typename Element, typename std::enable_if_t<is_duration<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    CUDF_FAIL("duration printing not supported yet");
  }

  template <typename Element,
            typename std::enable_if_t<std::is_same<Element, cudf::list_view>::value>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    lists_column_view lcv(col);

    std::string tmp =
      get_nested_type_str(col) + ":\n" + indent + "Length : " + std::to_string(lcv.size()) + "\n" +
      indent + "Offsets : " + (lcv.size() > 0 ? to_string(lcv.offsets(), ", ") : "") + "\n" +
      (lcv.has_nulls() ? indent + "Null count: " + std::to_string(lcv.null_count()) + "\n" +
                           detail::to_string(bitmask_to_host(col), col.size(), indent) + "\n"
                       : "") +
      indent + "Children :\n" +
      (lcv.size() > 0 ? detail::to_string(lcv.child(), ", ", indent + "   ") : "") + "\n";

    out.push_back(tmp);
  }
};

}  // namespace

namespace detail {

/**
 * @copydoc cudf::test::detail::to_strings
 *
 */
std::vector<std::string> to_strings(cudf::column_view const& col, std::string const& indent)
{
  std::vector<std::string> reply;
  cudf::type_dispatcher(col.type(), column_view_printer{}, col, reply, indent);
  return reply;
}

/**
 * @copydoc cudf::test::detail::to_string(cudf::column_view, std::string, std::string)
 *
 * @param indent Indentation for all output
 */
std::string to_string(cudf::column_view const& col,
                      std::string const& delimiter,
                      std::string const& indent)
{
  std::ostringstream buffer;
  std::vector<std::string> h_data = to_strings(col, indent);

  buffer << indent;
  std::copy(h_data.begin(),
            h_data.end() - (!h_data.empty()),
            std::ostream_iterator<std::string>(buffer, delimiter.c_str()));
  if (!h_data.empty()) buffer << h_data.back();

  return buffer.str();
}

/**
 * @copydoc cudf::test::detail::to_string(std::vector<bitmask_type>, size_type, std::string)
 *
 * @param indent Indentation for all output.  See comment in `to_strings` for
 * a detailed description.
 */
std::string to_string(std::vector<bitmask_type> const& null_mask,
                      size_type null_mask_size,
                      std::string const& indent)
{
  std::ostringstream buffer;
  buffer << indent;
  for (int idx = null_mask_size - 1; idx >= 0; idx--) {
    buffer << (cudf::bit_is_set(null_mask.data(), idx) ? "1" : "0");
  }
  return buffer.str();
}

}  // namespace detail

/**
 * @copydoc cudf::test::to_strings
 *
 */
std::vector<std::string> to_strings(cudf::column_view const& col)
{
  return detail::to_strings(col);
}

/**
 * @copydoc cudf::test::to_string(cudf::column_view, std::string)
 *
 */
std::string to_string(cudf::column_view const& col, std::string const& delimiter)
{
  return detail::to_string(col, delimiter);
}

/**
 * @copydoc cudf::test::to_string(std::vector<bitmask_type>, size_type)
 *
 */
std::string to_string(std::vector<bitmask_type> const& null_mask, size_type null_mask_size)
{
  return detail::to_string(null_mask, null_mask_size);
}

/**
 * @copydoc cudf::test::print
 *
 */
void print(cudf::column_view const& col, std::ostream& os, std::string const& delimiter)
{
  os << to_string(col, delimiter) << std::endl;
}

/**
 * @copydoc cudf::test::validate_host_masks
 *
 */
bool validate_host_masks(std::vector<bitmask_type> const& expected_mask,
                         std::vector<bitmask_type> const& got_mask,
                         size_type number_of_elements)
{
  return std::all_of(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(number_of_elements),
                     [&expected_mask, &got_mask](auto index) {
                       return cudf::bit_is_set(expected_mask.data(), index) ==
                              cudf::bit_is_set(got_mask.data(), index);
                     });
}

}  // namespace test
}  // namespace cudf
