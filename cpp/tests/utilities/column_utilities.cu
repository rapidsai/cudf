/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/detail/column_utilities.hpp>

#include <jit/type.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <numeric>
#include <sstream>
#include "cudf/detail/utilities/vector_factories.hpp"
#include "rmm/cuda_stream_view.hpp"

namespace cudf {

namespace test {

namespace {

// expand all non-null rows in a list column into a column of child row indices.
std::unique_ptr<column> generate_child_row_indices(lists_column_view const& c,
                                                   column_view const& row_indices)
{
  // Example input
  // List<int32_t>:
  // Length : 7
  // Offsets : 0, 3, 6, 8, 11, 14, 16, 19
  //                 |     |                        <-- non-null input rows
  // Null count: 5
  // 0010100
  //    1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7
  //                      |  |           |  |  |    <-- child rows of non-null rows
  //
  // Desired output:  [6, 7, 11, 12, 13]

  // compute total # of child row indices we will be emitting.
  auto row_size_iter = cudf::detail::make_counting_transform_iterator(
    0,
    [row_indices = row_indices.begin<size_type>(),
     validity    = c.null_mask(),
     offsets     = c.offsets().begin<offset_type>(),
     offset      = c.offset()] __device__(int index) {
      // both null mask and offsets data are not pre-sliced. so we need to add the column offset to
      // every incoming index.
      auto const true_index = row_indices[index] + offset;
      return !validity || cudf::bit_is_set(validity, true_index)
               ? (offsets[true_index + 1] - offsets[true_index])
               : 0;
    });
  auto const output_size =
    thrust::reduce(rmm::exec_policy(), row_size_iter, row_size_iter + row_indices.size());
  // no output. done.
  auto result =
    cudf::make_fixed_width_column(data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED);
  if (output_size == 0) { return result; }

  // for all input rows, what position in the output column they will start at.
  //
  // output_row_start = [0, 0, 0, 2, 2, 5, 5]
  //                           |     |              <-- non-null input rows
  //
  auto output_row_start = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, row_indices.size(), mask_state::UNALLOCATED);
  thrust::exclusive_scan(rmm::exec_policy(),
                         row_size_iter,
                         row_size_iter + row_indices.size(),
                         output_row_start->mutable_view().begin<size_type>());

  // fill result column with 1s
  //
  // result = [1, 1, 1, 1, 1]
  //
  thrust::generate(rmm::exec_policy(),
                   result->mutable_view().begin<size_type>(),
                   result->mutable_view().end<size_type>(),
                   [] __device__() { return 1; });

  // scatter the output row positions into result buffer
  //
  // result = [6, 1, 11, 1, 1]
  //
  auto validity_iter = cudf::detail::make_counting_transform_iterator(
    0,
    [row_indices = row_indices.begin<size_type>(),
     validity    = c.null_mask(),
     offset      = c.offset()] __device__(int index) {
      auto const true_index = row_indices[index] + offset;
      return !validity || cudf::bit_is_set(validity, true_index) ? 1 : 0;
    });
  auto output_row_iter = cudf::detail::make_counting_transform_iterator(
    0,
    [row_indices  = row_indices.begin<size_type>(),
     offsets      = c.offsets().begin<offset_type>(),
     offset       = c.offset(),
     first_offset = cudf::detail::get_value<offset_type>(
       c.offsets(), c.offset(), rmm::cuda_stream_default)] __device__(int index) {
      auto const true_index = row_indices[index] + offset;
      return offsets[true_index] - first_offset;
    });
  thrust::scatter_if(rmm::exec_policy(),
                     output_row_iter,
                     output_row_iter + row_indices.size(),
                     output_row_start->view().begin<size_type>(),
                     validity_iter,
                     result->mutable_view().begin<size_type>());

  // generate keys for each output row
  //
  // result = [1, 1, 2, 2, 2]
  //
  auto keys =
    cudf::make_fixed_width_column(data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED);
  thrust::generate(rmm::exec_policy(),
                   keys->mutable_view().begin<size_type>(),
                   keys->mutable_view().end<size_type>(),
                   [] __device__() { return 0; });
  thrust::scatter_if(rmm::exec_policy(),
                     validity_iter,
                     validity_iter + row_indices.size(),
                     output_row_start->view().begin<size_type>(),
                     validity_iter,
                     keys->mutable_view().begin<size_type>());
  thrust::inclusive_scan(rmm::exec_policy(),
                         keys->view().begin<size_type>(),
                         keys->view().end<size_type>(),
                         keys->mutable_view().begin<size_type>());

  // scan by key to generate final child row indices.
  // input
  //    result = [6, 1, 11, 1, 1]
  //    keys   = [1, 1, 2,  2, 2]
  //
  // output
  //    result = [6, 7, 11, 12, 13]
  //
  thrust::inclusive_scan_by_key(rmm::exec_policy(),
                                keys->view().begin<size_type>(),
                                keys->view().end<size_type>(),
                                result->view().begin<size_type>(),
                                result->mutable_view().begin<size_type>());
  return result;
}

#define PROP_EXPECT_EQ(a, b, _msg) \
  do {                             \
    if (throw_on_fail) {           \
      CUDF_EXPECTS(a == b, _msg);  \
    } else {                       \
      EXPECT_EQ(a, b);             \
    }                              \
  } while (0)

template <bool check_exact_equality>
struct column_property_comparator {
  bool types_equivalent(cudf::data_type const& lhs, cudf::data_type const& rhs)
  {
    return is_fixed_point(lhs) ? lhs.id() == rhs.id() : lhs == rhs;
  }

  size_type count_nulls(cudf::column_view const& c, cudf::column_view const& row_indices)
  {
    auto validity_iter = cudf::detail::make_counting_transform_iterator(
      0,
      [row_indices = row_indices.begin<size_type>(),
       validity    = c.null_mask(),
       offset      = c.offset()] __device__(int index) {
        // both null mask and offsets data are not pre-sliced. so we need to add the column offset
        // to every incoming index.
        auto const true_index = row_indices[index] + offset;
        return !validity || cudf::bit_is_set(validity, true_index) ? 0 : 1;
      });
    return thrust::reduce(rmm::exec_policy(), validity_iter, validity_iter + row_indices.size());
  }

  void compare_common(cudf::column_view const& lhs,
                      cudf::column_view const& rhs,
                      cudf::column_view const& lhs_row_indices,
                      cudf::column_view const& rhs_row_indices,
                      bool throw_on_fail)
  {
    if (check_exact_equality) {
      PROP_EXPECT_EQ(lhs.type(), rhs.type(), "Type mismatch");
    } else {
      PROP_EXPECT_EQ(types_equivalent(lhs.type(), rhs.type()), true, "Type mismatch");
    }

    // DISCUSSION: does this make sense, semantically?
    auto const lhs_size = check_exact_equality ? lhs.size() : lhs_row_indices.size();
    auto const rhs_size = check_exact_equality ? rhs.size() : rhs_row_indices.size();
    PROP_EXPECT_EQ(lhs_size, rhs_size, "Column size mismatch");

    if (lhs_size > 0 && check_exact_equality) {
      PROP_EXPECT_EQ(lhs.nullable(), rhs.nullable(), "Column nullability mismatch");
    }

    // DISCUSSION: does this make sense, semantically?
    auto const lhs_null_count =
      check_exact_equality ? lhs.null_count() : count_nulls(lhs, lhs_row_indices);
    auto const rhs_null_count =
      check_exact_equality ? rhs.null_count() : count_nulls(rhs, rhs_row_indices);
    PROP_EXPECT_EQ(lhs_null_count, rhs_null_count, "Null count mismatch");

    // equivalent, but not exactly equal columns can have a different number of children if their
    // sizes are both 0. Specifically, empty string columns may or may not have children.
    if (check_exact_equality || (lhs.size() > 0 && lhs.null_count() < lhs.size())) {
      PROP_EXPECT_EQ(lhs.num_children(), rhs.num_children(), "Child count mismatch");
    }
  }

  template <typename T,
            std::enable_if_t<!std::is_same<T, cudf::list_view>::value &&
                             !std::is_same<T, cudf::struct_view>::value>* = nullptr>
  void operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  bool throw_on_fail)
  {
    compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, throw_on_fail);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::list_view>::value>* = nullptr>
  void operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  bool throw_on_fail)
  {
    compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, throw_on_fail);

    cudf::lists_column_view lhs_l(lhs);
    cudf::lists_column_view rhs_l(rhs);

    // recurse
    auto lhs_child = lhs_l.get_sliced_child(rmm::cuda_stream_default);
    // note: if a column is all nulls or otherwise empty, no indices are generated and no recursion
    // happens
    auto lhs_child_indices = generate_child_row_indices(lhs_l, lhs_row_indices);
    if (lhs_child_indices->size() > 0) {
      auto rhs_child         = rhs_l.get_sliced_child(rmm::cuda_stream_default);
      auto rhs_child_indices = generate_child_row_indices(rhs_l, rhs_row_indices);
      cudf::type_dispatcher(lhs_child.type(),
                            column_property_comparator<check_exact_equality>{},
                            lhs_child,
                            rhs_child,
                            *lhs_child_indices,
                            *rhs_child_indices,
                            throw_on_fail);
    }
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::struct_view>::value>* = nullptr>
  void operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  bool throw_on_fail)
  {
    compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, throw_on_fail);

    structs_column_view l_scv(lhs);
    structs_column_view r_scv(rhs);

    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(lhs.num_children()),
                  [&](auto i) {
                    column_view lhs_child = l_scv.get_sliced_child(i);
                    column_view rhs_child = r_scv.get_sliced_child(i);
                    cudf::type_dispatcher(lhs_child.type(),
                                          column_property_comparator<check_exact_equality>{},
                                          lhs_child,
                                          rhs_child,
                                          lhs_row_indices,
                                          rhs_row_indices,
                                          throw_on_fail);
                  });
  }
};

class corresponding_rows_unequal {
 public:
  corresponding_rows_unequal(table_device_view d_lhs,
                             table_device_view d_rhs,
                             column_device_view lhs_row_indices_,
                             column_device_view rhs_row_indices_)
    : comp(d_lhs, d_rhs), lhs_row_indices(lhs_row_indices_), rhs_row_indices(rhs_row_indices_)
  {
  }

  cudf::row_equality_comparator<true> comp;

  __device__ bool operator()(size_type index)
  {
    return !comp(lhs_row_indices.element<size_type>(index),
                 rhs_row_indices.element<size_type>(index));
  }

  column_device_view lhs_row_indices;
  column_device_view rhs_row_indices;
};

class corresponding_rows_not_equivalent {
  table_device_view d_lhs;
  table_device_view d_rhs;

  column_device_view lhs_row_indices;
  column_device_view rhs_row_indices;

 public:
  corresponding_rows_not_equivalent(table_device_view d_lhs,
                                    table_device_view d_rhs,
                                    column_device_view lhs_row_indices_,
                                    column_device_view rhs_row_indices_)
    : d_lhs(d_lhs),
      d_rhs(d_rhs),
      comp(d_lhs, d_rhs),
      lhs_row_indices(lhs_row_indices_),
      rhs_row_indices(rhs_row_indices_)
  {
    CUDF_EXPECTS(d_lhs.num_columns() == 1 and d_rhs.num_columns() == 1,
                 "Unsupported number of columns");
  }

  struct typed_element_not_equivalent {
    template <typename T>
    __device__ std::enable_if_t<std::is_floating_point<T>::value, bool> operator()(
      column_device_view const& lhs,
      column_device_view const& rhs,
      size_type lhs_index,
      size_type rhs_index)
    {
      if (lhs.is_valid(lhs_index) and rhs.is_valid(rhs_index)) {
        T const x = lhs.element<T>(lhs_index);
        T const y = rhs.element<T>(rhs_index);

        // Must handle inf and nan separately
        if (std::isinf(x) || std::isinf(y)) {
          return x != y;  // comparison of (inf==inf) returns true
        } else if (std::isnan(x) || std::isnan(y)) {
          return std::isnan(x) != std::isnan(y);  // comparison of (nan==nan) returns false
        } else {
          constexpr int ulp     = 4;  // ulp = unit of least precision, value taken from google test
          T const abs_x_minus_y = std::abs(x - y);
          return abs_x_minus_y >= std::numeric_limits<T>::min() &&
                 abs_x_minus_y > std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp;
        }
      } else {
        // if either is null, then the inequality was checked already
        return true;
      }
    }

    template <typename T, typename... Args>
    __device__ std::enable_if_t<not std::is_floating_point<T>::value, bool> operator()(Args...)
    {
      // Non-floating point inequality is checked already
      return true;
    }
  };

  cudf::row_equality_comparator<true> comp;

  __device__ bool operator()(size_type index)
  {
    auto const lhs_index = lhs_row_indices.element<size_type>(index);
    auto const rhs_index = rhs_row_indices.element<size_type>(index);

    if (not comp(lhs_index, rhs_index)) {
      auto lhs_col = this->d_lhs.column(0);
      auto rhs_col = this->d_rhs.column(0);
      return type_dispatcher(
        lhs_col.type(), typed_element_not_equivalent{}, lhs_col, rhs_col, lhs_index, rhs_index);
    }
    return false;
  }
};

// Stringify the inconsistent values resulted from the comparison of two columns element-wise
std::string stringify_column_differences(cudf::device_span<int const> differences,
                                         column_view const& lhs,
                                         column_view const& rhs,
                                         column_view const& lhs_row_indices,
                                         column_view const& rhs_row_indices,
                                         bool print_all_differences,
                                         int depth)
{
  CUDF_EXPECTS(not differences.empty(), "Shouldn't enter this function if `differences` is empty");
  std::string const depth_str = depth > 0 ? "depth " + std::to_string(depth) + '\n' : "";
  // move the differences to the host.
  auto h_differences = cudf::detail::make_host_vector_sync(differences);
  if (print_all_differences) {
    std::ostringstream buffer;
    buffer << depth_str << "differences:" << std::endl;

    auto source_table = cudf::table_view({lhs, rhs});
    auto diff_column =
      fixed_width_column_wrapper<int32_t>(h_differences.begin(), h_differences.end());
    auto diff_table = cudf::gather(source_table, diff_column);
    //  Need to pull back the differences
    auto const h_left_strings = to_strings(diff_table->get_column(0));

    auto const h_right_strings = to_strings(diff_table->get_column(1));
    for (size_t i = 0; i < h_differences.size(); ++i)
      buffer << depth_str << "lhs[" << h_differences[i] << "] = " << h_left_strings[i] << ", rhs["
             << h_differences[i] << "] = " << h_right_strings[i] << std::endl;
    return buffer.str();
  } else {
    auto const index = h_differences[0];  // only stringify first difference

    auto const lhs_index =
      cudf::detail::get_value<size_type>(lhs_row_indices, index, rmm::cuda_stream_default);
    auto const rhs_index =
      cudf::detail::get_value<size_type>(rhs_row_indices, index, rmm::cuda_stream_default);
    auto diff_lhs = cudf::detail::slice(lhs, lhs_index, lhs_index + 1);
    auto diff_rhs = cudf::detail::slice(rhs, rhs_index, rhs_index + 1);
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
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  bool print_all_differences,
                  bool throw_on_fail,
                  int depth)
  {
    auto d_lhs = cudf::table_device_view::create(table_view{{lhs}});
    auto d_rhs = cudf::table_device_view::create(table_view{{rhs}});

    auto d_lhs_row_indices = cudf::column_device_view::create(lhs_row_indices);
    auto d_rhs_row_indices = cudf::column_device_view::create(rhs_row_indices);

    using ComparatorType = std::conditional_t<check_exact_equality,
                                              corresponding_rows_unequal,
                                              corresponding_rows_not_equivalent>;

    auto differences = rmm::device_uvector<int>(
      lhs.size(), rmm::cuda_stream_default);  // worst case: everything different
    auto input_iter = thrust::make_counting_iterator(0);
    auto diff_iter =
      thrust::copy_if(rmm::exec_policy(),
                      input_iter,
                      input_iter + lhs_row_indices.size(),
                      differences.begin(),
                      ComparatorType(*d_lhs, *d_rhs, *d_lhs_row_indices, *d_rhs_row_indices));

    differences.resize(thrust::distance(differences.begin(), diff_iter),
                       rmm::cuda_stream_default);  // shrink back down

    if (not differences.is_empty()) {
      if (throw_on_fail) {
        CUDF_FAIL("Column differences detected");
      } else {
        GTEST_FAIL() << stringify_column_differences(
          differences, lhs, rhs, lhs_row_indices, rhs_row_indices, print_all_differences, depth);
      }
    }
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
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  bool print_all_differences,
                  bool throw_on_fail,
                  int depth)
  {
    lists_column_view lhs_l(lhs);
    lists_column_view rhs_l(rhs);

    CUDF_EXPECTS(lhs_row_indices.size() == rhs_row_indices.size(), "List column size mismatch");
    if (lhs_row_indices.is_empty()) { return; }

    // worst case - everything is different
    rmm::device_uvector<int> differences(lhs_row_indices.size(), rmm::cuda_stream_default);

    // compare offsets, taking slicing into account

    // left side
    size_type lhs_shift =
      cudf::detail::get_value<size_type>(lhs_l.offsets(), lhs_l.offset(), rmm::cuda_stream_default);
    auto lhs_offsets = thrust::make_transform_iterator(
      lhs_l.offsets().begin<size_type>() + lhs_l.offset(),
      [lhs_shift] __device__(size_type offset) { return offset - lhs_shift; });
    auto lhs_valids = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [mask = lhs_l.null_mask(), offset = lhs_l.offset()] __device__(size_type index) {
        return mask == nullptr ? true : cudf::bit_is_set(mask, index + offset);
      });

    // right side
    size_type rhs_shift =
      cudf::detail::get_value<size_type>(rhs_l.offsets(), rhs_l.offset(), rmm::cuda_stream_default);
    auto rhs_offsets = thrust::make_transform_iterator(
      rhs_l.offsets().begin<size_type>() + rhs_l.offset(),
      [rhs_shift] __device__(size_type offset) { return offset - rhs_shift; });
    auto rhs_valids = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [mask = rhs_l.null_mask(), offset = rhs_l.offset()] __device__(size_type index) {
        return mask == nullptr ? true : cudf::bit_is_set(mask, index + offset);
      });

    // when checking for equivalency, we can't compare offset values directly, we can only
    // compare lengths of the rows, and only if valid.  as a concrete example, you could have two
    // equivalent columns with the following data:
    //
    // column A
    //    offsets =  [0, 3, 5, 7]
    //    validity = [0, 1, 1, 1]
    //
    // column B
    //   offsets =   [0, 0, 2, 4]
    //   validity =  [0, 1, 1, 1]
    //
    // Row 0 in column A happens to have a positive length, even though the row is null, but column
    // B does not.  So the offsets for the remaining valid rows are fundamentally different even
    // though the row lengths are the same.
    //
    auto input_iter = thrust::make_counting_iterator(0);
    auto diff_iter  = thrust::copy_if(
      rmm::exec_policy(),
      input_iter,
      input_iter + lhs_row_indices.size(),
      differences.begin(),
      [lhs_offsets,
       rhs_offsets,
       lhs_valids,
       rhs_valids,
       lhs_indices = lhs_row_indices.begin<size_type>(),
       rhs_indices = rhs_row_indices.begin<size_type>()] __device__(size_type index) {
        auto const lhs_index = lhs_indices[index];
        auto const rhs_index = rhs_indices[index];

        // check for validity match
        if (lhs_valids[lhs_index] != rhs_valids[rhs_index]) { return true; }

        // if the row is valid, check that the length of the list is the same. do this
        // for both the equivalency and exact equality checks.
        if (lhs_valids[lhs_index] && ((lhs_offsets[lhs_index + 1] - lhs_offsets[lhs_index]) !=
                                      (rhs_offsets[rhs_index + 1] - rhs_offsets[rhs_index]))) {
          return true;
        }

        // if validity matches -and- is false, we can ignore the actual offset values. this
        // is technically not checking "equal()", but it's how the non-list code path handles it
        if (!lhs_valids[lhs_index]) { return false; }

        // if checking exact equality, compare the actual offset values
        if (check_exact_equality && lhs_offsets[lhs_index] != rhs_offsets[rhs_index]) {
          return true;
        }

        return false;
      });

    differences.resize(thrust::distance(differences.begin(), diff_iter),
                       rmm::cuda_stream_default);  // shrink back down

    if (not differences.is_empty()) {
      if (throw_on_fail) {
        CUDF_FAIL("Column differences detected");
      } else {
        GTEST_FAIL() << stringify_column_differences(
          differences, lhs, rhs, lhs_row_indices, rhs_row_indices, print_all_differences, depth);
      }
    }

    // recurse.
    auto lhs_child = lhs_l.get_sliced_child(rmm::cuda_stream_default);
    // note: if a column is all nulls or otherwise empty, no indices are generated and no recursion
    // happens
    auto lhs_child_indices = generate_child_row_indices(lhs_l, lhs_row_indices);
    if (lhs_child_indices->size() > 0) {
      auto rhs_child         = rhs_l.get_sliced_child(rmm::cuda_stream_default);
      auto rhs_child_indices = generate_child_row_indices(rhs_l, rhs_row_indices);
      cudf::type_dispatcher(lhs_child.type(),
                            column_comparator<check_exact_equality>{},
                            lhs_child,
                            rhs_child,
                            *lhs_child_indices,
                            *rhs_child_indices,
                            print_all_differences,
                            throw_on_fail,
                            depth + 1);
    }
  }
};

template <bool check_exact_equality>
struct column_comparator_impl<struct_view, check_exact_equality> {
  void operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  bool print_all_differences,
                  bool throw_on_fail,
                  int depth)
  {
    structs_column_view l_scv(lhs);
    structs_column_view r_scv(rhs);

    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + lhs.num_children(),
                  [&](auto i) {
                    column_view lhs_child = l_scv.get_sliced_child(i);
                    column_view rhs_child = r_scv.get_sliced_child(i);
                    cudf::type_dispatcher(lhs_child.type(),
                                          column_comparator<check_exact_equality>{},
                                          lhs_child,
                                          rhs_child,
                                          lhs_row_indices,
                                          rhs_row_indices,
                                          print_all_differences,
                                          throw_on_fail,
                                          depth + 1);
                  });
  }
};

template <bool check_exact_equality>
struct column_comparator {
  template <typename T>
  void operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  bool print_all_differences = false,
                  bool throw_on_fail         = false,
                  int depth                  = 0)
  {
    CUDF_EXPECTS(lhs_row_indices.size() == rhs_row_indices.size(),
                 "Mismatch in row counts to compare");

    // compare properties
    cudf::type_dispatcher(lhs.type(),
                          column_property_comparator<check_exact_equality>{},
                          lhs,
                          rhs,
                          lhs_row_indices,
                          rhs_row_indices,
                          throw_on_fail);

    // compare values
    column_comparator_impl<T, check_exact_equality> comparator{};
    comparator(
      lhs, rhs, lhs_row_indices, rhs_row_indices, print_all_differences, throw_on_fail, depth);
  }
};

std::unique_ptr<column> generate_all_row_indices(size_type num_rows)
{
  auto indices =
    cudf::make_fixed_width_column(data_type{type_id::INT32}, num_rows, mask_state::UNALLOCATED);
  thrust::sequence(rmm::exec_policy(),
                   indices->mutable_view().begin<size_type>(),
                   indices->mutable_view().end<size_type>(),
                   0);
  return indices;
}

}  // namespace

/**
 * @copydoc cudf::test::expect_column_properties_equal
 */
void expect_column_properties_equal(column_view const& lhs,
                                    column_view const& rhs,
                                    bool throw_on_fail)
{
  auto indices = generate_all_row_indices(lhs.size());
  cudf::type_dispatcher(
    lhs.type(), column_property_comparator<true>{}, lhs, rhs, *indices, *indices, throw_on_fail);
}

/**
 * @copydoc cudf::test::expect_column_properties_equivalent
 */
void expect_column_properties_equivalent(column_view const& lhs,
                                         column_view const& rhs,
                                         bool throw_on_fail)
{
  auto indices = generate_all_row_indices(lhs.size());
  cudf::type_dispatcher(
    lhs.type(), column_property_comparator<false>{}, lhs, rhs, *indices, *indices, throw_on_fail);
}

/**
 * @copydoc cudf::test::expect_columns_equal
 */
void expect_columns_equal(cudf::column_view const& lhs,
                          cudf::column_view const& rhs,
                          bool print_all_differences,
                          bool throw_on_fail)
{
  auto indices = generate_all_row_indices(lhs.size());
  cudf::type_dispatcher(lhs.type(),
                        column_comparator<true>{},
                        lhs,
                        rhs,
                        *indices,
                        *indices,
                        print_all_differences,
                        throw_on_fail);
}

/**
 * @copydoc cudf::test::expect_columns_equivalent
 */
void expect_columns_equivalent(cudf::column_view const& lhs,
                               cudf::column_view const& rhs,
                               bool print_all_differences,
                               bool throw_on_fail)
{
  auto indices = generate_all_row_indices(lhs.size());
  cudf::type_dispatcher(lhs.type(),
                        column_comparator<false>{},
                        lhs,
                        rhs,
                        *indices,
                        *indices,
                        print_all_differences,
                        throw_on_fail);
}

/**
 * @copydoc cudf::test::expect_equal_buffers
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
  size_type first =
    cudf::detail::get_value<size_type>(offsets, c.offset(), rmm::cuda_stream_default);
  rmm::device_uvector<size_type> shifted_offsets(output_size, rmm::cuda_stream_default);

  // normalize the offset values for the column offset
  size_type const* d_offsets = offsets.head<size_type>() + c.offset();
  thrust::transform(
    rmm::exec_policy(),
    d_offsets,
    d_offsets + output_size,
    shifted_offsets.begin(),
    [first] __device__(int32_t offset) { return static_cast<size_type>(offset - first); });

  auto const h_shifted_offsets = cudf::detail::make_host_vector_sync(shifted_offsets);
  std::ostringstream buffer;
  for (size_t idx = 0; idx < h_shifted_offsets.size(); idx++) {
    buffer << h_shifted_offsets[idx];
    if (idx < h_shifted_offsets.size() - 1) { buffer << delimiter; }
  }
  return buffer.str();
}

struct column_view_printer {
  template <typename Element, typename std::enable_if_t<is_numeric<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
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
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
  {
    auto const h_data = cudf::test::to_host<Element>(col);
    if (col.nullable()) {
      std::transform(thrust::make_counting_iterator(size_type{0}),
                     thrust::make_counting_iterator(col.size()),
                     std::back_inserter(out),
                     [&h_data](auto idx) {
                       return h_data.second.empty() || bit_is_set(h_data.second.data(), idx)
                                ? static_cast<std::string>(h_data.first[idx])
                                : std::string("NULL");
                     });
    } else {
      std::transform(std::cbegin(h_data.first),
                     std::cend(h_data.first),
                     std::back_inserter(out),
                     [col](auto const& fp) { return static_cast<std::string>(fp); });
    }
  }

  template <typename Element,
            typename std::enable_if_t<std::is_same<Element, cudf::string_view>::value>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
  {
    //
    //  Implementation for strings, call special to_host variant
    //
    if (col.is_empty()) return;
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
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
  {
    cudf::dictionary_column_view dictionary(col);
    if (col.is_empty()) return;
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
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
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

    // propagate slicing to the child if necessary
    column_view child    = lcv.get_sliced_child(rmm::cuda_stream_default);
    bool const is_sliced = lcv.offset() > 0 || child.offset() > 0;

    std::string tmp =
      get_nested_type_str(col) + (is_sliced ? "(sliced)" : "") + ":\n" + indent +
      "Length : " + std::to_string(lcv.size()) + "\n" + indent +
      "Offsets : " + (lcv.size() > 0 ? nested_offsets_to_string(lcv) : "") + "\n" +
      (lcv.parent().nullable()
         ? indent + "Null count: " + std::to_string(lcv.null_count()) + "\n" +
             detail::to_string(bitmask_to_host(col), col.size(), indent) + "\n"
         : "") +
      // non-nested types don't typically display their null masks, so do it here for convenience.
      (!is_nested(child.type()) && child.nullable()
         ? "   " + detail::to_string(bitmask_to_host(child), child.size(), indent) + "\n"
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
    if (view.nullable()) {
      out_stream << indent << "Null count: " << view.null_count() << "\n"
                 << detail::to_string(bitmask_to_host(col), col.size(), indent) << "\n";
    }

    auto iter = thrust::make_counting_iterator(0);
    std::transform(
      iter,
      iter + view.num_children(),
      std::ostream_iterator<std::string>(out_stream, "\n"),
      [&](size_type index) {
        auto child = view.get_sliced_child(index);

        // non-nested types don't typically display their null masks, so do it here for convenience.
        return (!is_nested(child.type()) && child.nullable()
                  ? "   " + detail::to_string(bitmask_to_host(child), child.size(), indent) + "\n"
                  : "") +
               detail::to_string(child, ", ", indent + "   ");
      });

    out.push_back(out_stream.str());
  }
};

}  // namespace

namespace detail {

/**
 * @copydoc cudf::test::detail::to_strings
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
 */
std::vector<std::string> to_strings(cudf::column_view const& col)
{
  return detail::to_strings(col);
}

/**
 * @copydoc cudf::test::to_string(cudf::column_view, std::string)
 */
std::string to_string(cudf::column_view const& col, std::string const& delimiter)
{
  return detail::to_string(col, delimiter);
}

/**
 * @copydoc cudf::test::to_string(std::vector<bitmask_type>, size_type)
 */
std::string to_string(std::vector<bitmask_type> const& null_mask, size_type null_mask_size)
{
  return detail::to_string(null_mask, null_mask_size);
}

/**
 * @copydoc cudf::test::print
 */
void print(cudf::column_view const& col, std::ostream& os, std::string const& delimiter)
{
  os << to_string(col, delimiter) << std::endl;
}

/**
 * @copydoc cudf::test::validate_host_masks
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
