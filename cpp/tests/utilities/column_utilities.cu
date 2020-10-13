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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/bit.hpp>
#include "cudf/utilities/type_dispatcher.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/detail/column_utilities.hpp>

#include <jit/type.h>

#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <numeric>
#include <sstream>

namespace cudf {
namespace test {

namespace {

template <bool check_exact_equality>
struct column_property_comparator {
  void compare_common(cudf::column_view const& lhs, cudf::column_view const& rhs)
  {
    EXPECT_EQ(lhs.type(), rhs.type());
    EXPECT_EQ(lhs.size(), rhs.size());
    if (lhs.size() > 0 && check_exact_equality) { EXPECT_EQ(lhs.nullable(), rhs.nullable()); }

    // equivalent, but not exactly equal columns can have a different number of children if their
    // sizes are both 0. Specifically, empty string columns may or may not have children.
    if (check_exact_equality || lhs.size() > 0) {
      EXPECT_EQ(lhs.num_children(), rhs.num_children());
    }
  }

  template <typename T, std::enable_if_t<!std::is_same<T, cudf::list_view>::value>* = nullptr>
  void operator()(cudf::column_view const& lhs, cudf::column_view const& rhs)
  {
    compare_common(lhs, rhs);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::list_view>::value>* = nullptr>
  void operator()(cudf::column_view const& lhs, cudf::column_view const& rhs)
  {
    compare_common(lhs, rhs);

    cudf::lists_column_view lhs_l(lhs);
    cudf::lists_column_view rhs_l(rhs);

    // recurse
    cudf::type_dispatcher(lhs_l.child().type(),
                          column_property_comparator<check_exact_equality>{},
                          lhs_l.get_sliced_child(0),
                          rhs_l.get_sliced_child(0));
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

std::string differences_message(thrust::device_vector<int> const& differences,
                                column_view const& lhs,
                                column_view const& rhs,
                                bool all_differences,
                                int depth)
{
  CUDF_EXPECTS(not differences.empty(), "Shouldn't enter this function if `differences` is empty");

  std::string const depth_str = depth > 0 ? "depth " + std::to_string(depth) + '\n' : "";

  if (all_differences) {
    std::ostringstream buffer;
    buffer << depth_str << "differences:" << std::endl;

    auto source_table = cudf::table_view({lhs, rhs});
    auto diff_column  = fixed_width_column_wrapper<int32_t>(differences.begin(), differences.end());
    auto diff_table   = cudf::gather(source_table, diff_column);

    //  Need to pull back the differences
    auto const h_left_strings  = to_strings(diff_table->get_column(0));
    auto const h_right_strings = to_strings(diff_table->get_column(1));

    for (size_t i = 0; i < differences.size(); ++i)
      buffer << depth_str << "lhs[" << differences[i] << "] = " << h_left_strings[i] << ", rhs["
             << differences[i] << "] = " << h_right_strings[i] << std::endl;

    return buffer.str();
  } else {
    int index = differences[0];  // only stringify first difference

    auto diff_lhs = cudf::detail::slice(lhs, index, index + 1);
    auto diff_rhs = cudf::detail::slice(rhs, index, index + 1);

    return depth_str + "first difference: " + "lhs[" + std::to_string(index) +
           "] = " + to_string(diff_lhs, "") + ", rhs[" + std::to_string(index) +
           "] = " + to_string(diff_rhs, "");
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
    auto d_lhs = cudf::table_device_view::create(table_view{{lhs}});
    auto d_rhs = cudf::table_device_view::create(table_view{{rhs}});

    using ComparatorType = std::conditional_t<check_exact_equality,
                                              corresponding_rows_unequal,
                                              corresponding_rows_not_equivalent>;

    auto differences = thrust::device_vector<int>(lhs.size());  // worst case: everything different
    auto diff_iter   = thrust::copy_if(thrust::device,
                                     thrust::make_counting_iterator(0),
                                     thrust::make_counting_iterator(lhs.size()),
                                     differences.begin(),
                                     ComparatorType(*d_lhs, *d_rhs));

    differences.resize(thrust::distance(differences.begin(), diff_iter));  // shrink back down

    if (not differences.empty())
      GTEST_FAIL() << differences_message(differences, lhs, rhs, print_all_differences, depth);
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

    CUDF_EXPECTS(lhs_l.size() == rhs_l.size(), "List column size mismatch");
    if (lhs_l.size() == 0) { return; }

    // worst case - everything is different
    thrust::device_vector<int> differences(lhs.size());

    // TODO : determine how equals/equivalency should work for columns with divergent underlying
    // data, but equivalent null masks. Example:
    //
    // List<int32_t>:
    // Length : 3
    // Offsets : 0, 3, 5, 5
    // Nulls: 011
    // Children :
    //   1, 2, 3, 4, 5
    //
    // List<int32_t>:
    // Length : 3
    // Offsets : 0, 3, 5, 7
    // Nulls: 011
    // Children :
    //   1, 2, 3, 4, 5, 7, 8
    //
    // These two columns are seemingly equivalent, since their top level rows are the same, with
    // just the last element being null. However, pyArrow will say these are -not- equal and
    // does not appear to have an equivalent() check.  So the question is : should we be handling
    // this case when someone calls expect_columns_equivalent()?

    // compare offsets, taking slicing into account

    // left side
    size_type lhs_shift = cudf::detail::get_value<size_type>(lhs_l.offsets(), lhs_l.offset(), 0);
    auto lhs_offsets    = thrust::make_transform_iterator(
      lhs_l.offsets().begin<size_type>() + lhs_l.offset(),
      [lhs_shift] __device__(size_type offset) { return offset - lhs_shift; });
    auto lhs_valids = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [mask = lhs_l.null_mask(), offset = lhs_l.offset()] __device__(size_type index) {
        return mask == nullptr ? true : cudf::bit_is_set(mask, index + offset);
      });

    // right side
    size_type rhs_shift = cudf::detail::get_value<size_type>(rhs_l.offsets(), rhs_l.offset(), 0);
    auto rhs_offsets    = thrust::make_transform_iterator(
      rhs_l.offsets().begin<size_type>() + rhs_l.offset(),
      [rhs_shift] __device__(size_type offset) { return offset - rhs_shift; });
    auto rhs_valids = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [mask = rhs_l.null_mask(), offset = rhs_l.offset()] __device__(size_type index) {
        return mask == nullptr ? true : cudf::bit_is_set(mask, index + offset);
      });

    auto diff_iter = thrust::copy_if(
      thrust::device,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(lhs_l.size() + 1),
      differences.begin(),
      [lhs_offsets, rhs_offsets, lhs_valids, rhs_valids, num_rows = lhs_l.size()] __device__(
        size_type index) {
        // last offset has no validity associated with it
        if (index < num_rows - 1) {
          if (lhs_valids[index] != rhs_valids[index]) { return true; }
          // if validity matches -and- is false, we can ignore the actual values. this
          // is technically not checking "equal()", but it's how the non-list code path handles it
          if (!lhs_valids[index]) { return false; }
        }
        return lhs_offsets[index] == rhs_offsets[index] ? false : true;
      });

    differences.resize(thrust::distance(differences.begin(), diff_iter));  // shrink back down

    if (not differences.empty())
      GTEST_FAIL() << differences_message(differences, lhs, rhs, print_all_differences, depth);

    // recurse
    auto lhs_child = lhs_l.get_sliced_child(0);
    auto rhs_child = rhs_l.get_sliced_child(0);
    cudf::type_dispatcher(lhs_child.type(),
                          column_comparator<check_exact_equality>{},
                          lhs_child,
                          rhs_child,
                          print_all_differences,
                          depth + 1);
  }
};

template <bool check_exact_equality>
struct column_comparator_impl<struct_view, check_exact_equality> {
  void operator()(column_view const& lhs,
                  column_view const& rhs,
                  bool print_all_differences,
                  int depth)
  {
    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + lhs.num_children(),
                  [&](auto i) {
                    cudf::type_dispatcher(lhs.child(i).type(),
                                          column_comparator<check_exact_equality>{},
                                          lhs.child(i),
                                          rhs.child(i),
                                          print_all_differences,
                                          depth + 1);
                  });
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

static auto duration_suffix(cudf::duration_D) { return " days"; }

static auto duration_suffix(cudf::duration_s) { return " seconds"; }

static auto duration_suffix(cudf::duration_ms) { return " milliseconds"; }

static auto duration_suffix(cudf::duration_us) { return " microseconds"; }

static auto duration_suffix(cudf::duration_ns) { return " nanoseconds"; }

std::string get_nested_type_str(cudf::column_view const& view)
{
  if (view.type().id() == cudf::type_id::LIST) {
    lists_column_view lcv(view);
    return cudf::jit::get_type_name(view.type()) + "<" + (get_nested_type_str(lcv.child())) + ">";
  }

  if (view.type().id() == cudf::type_id::STRUCT) {
    std::ostringstream out;

    out << cudf::jit::get_type_name(view.type()) + "<";
    std::transform(view.child_begin(),
                   view.child_end(),
                   std::ostream_iterator<std::string>(out, ","),
                   [&out](auto const col) { return get_nested_type_str(col); });
    out << ">";
    return out.str();
  }

  return cudf::jit::get_type_name(view.type());
}

template <typename NestedColumnView>
std::string nested_offsets_to_string(NestedColumnView const& c, std::string const& delimiter = ", ")
{
  column_view offsets = (c.parent()).child(NestedColumnView::offsets_column_index);
  CUDF_EXPECTS(offsets.type().id() == type_id::INT32,
               "Column does not appear to be an offsets column");
  CUDF_EXPECTS(offsets.offset() == 0, "Offsets column has an internal offset!");
  size_type output_size = c.size() + 1;

  // the first offset value to normalize everything against
  size_type first = cudf::detail::get_value<size_type>(offsets, c.offset(), 0);
  rmm::device_vector<size_type> shifted_offsets(output_size);

  // normalize the offset values for the column offset
  size_type const* d_offsets = offsets.head<size_type>() + c.offset();
  thrust::transform(
    rmm::exec_policy(0)->on(0),
    d_offsets,
    d_offsets + output_size,
    shifted_offsets.begin(),
    [first] __device__(int32_t offset) { return static_cast<size_type>(offset - first); });

  thrust::host_vector<size_type> h_shifted_offsets(shifted_offsets);
  std::ostringstream buffer;
  for (size_t idx = 0; idx < h_shifted_offsets.size(); idx++) {
    buffer << h_shifted_offsets[idx];
    if (idx < h_shifted_offsets.size() - 1) { buffer << delimiter; }
  }
  return buffer.str();
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

  template <typename Element, typename std::enable_if_t<cudf::is_fixed_point<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    auto const h_data = cudf::test::to_host<Element>(col);
    std::transform(std::cbegin(h_data.first),
                   std::cend(h_data.first),
                   std::back_inserter(out),
                   [](auto const& fp) { return std::to_string(static_cast<double>(fp)); });
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
    std::vector<std::string> indices = to_strings({dictionary.indices().type(),
                                                   dictionary.size(),
                                                   dictionary.indices().head(),
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

  // Print the tick counts with the units
  template <typename Element, typename std::enable_if_t<is_duration<Element>()>* = nullptr>
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
                                ? numeric_to_string_precise(h_data.first[idx].count()) +
                                    duration_suffix(h_data.first[idx])
                                : std::string("NULL");
                     });

    } else {
      std::transform(h_data.first.begin(), h_data.first.end(), out.begin(), [](Element el) {
        return numeric_to_string_precise(el.count()) + duration_suffix(el);
      });
    }
  }

  template <typename Element,
            typename std::enable_if_t<std::is_same<Element, cudf::list_view>::value>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    lists_column_view lcv(col);

    // propage slicing to the child if necessary
    column_view child    = lcv.get_sliced_child(0);
    bool const is_sliced = lcv.offset() > 0 || child.offset() > 0;

    std::string tmp =
      get_nested_type_str(col) + (is_sliced ? "(sliced)" : "") + ":\n" + indent +
      "Length : " + std::to_string(lcv.size()) + "\n" + indent +
      "Offsets : " + (lcv.size() > 0 ? nested_offsets_to_string(lcv) : "") + "\n" +
      (lcv.has_nulls() ? indent + "Null count: " + std::to_string(lcv.null_count()) + "\n" +
                           detail::to_string(bitmask_to_host(col), col.size(), indent) + "\n"
                       : "") +
      indent + "Children :\n" +
      (child.type().id() != type_id::LIST && child.has_nulls()
         ? indent + detail::to_string(bitmask_to_host(child), child.size(), indent) + "\n"
         : "") +
      (detail::to_string(child, ", ", indent + "   ")) + "\n";

    out.push_back(tmp);
  }

  template <typename Element,
            typename std::enable_if_t<std::is_same<Element, cudf::struct_view>::value>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    structs_column_view view{col};

    std::ostringstream out_stream;

    out_stream << get_nested_type_str(col) << ":\n"
               << indent << "Length : " << view.size() << ":\n";
    if (view.has_nulls()) {
      out_stream << indent << "Null count: " << view.null_count() << "\n"
                 << detail::to_string(bitmask_to_host(col), col.size(), indent) << "\n";
    }

    std::transform(
      view.child_begin(),
      view.child_end(),
      std::ostream_iterator<std::string>(out_stream, "\n"),
      [&](auto child_column) { return detail::to_string(child_column, ", ", indent + "    "); });

    out.push_back(out_stream.str());
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
