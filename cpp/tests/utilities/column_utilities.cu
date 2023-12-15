/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuda/functional>

#include <numeric>
#include <sstream>

namespace cudf {

namespace test {

namespace {

std::unique_ptr<column> generate_all_row_indices(size_type num_rows)
{
  auto indices = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, num_rows, mask_state::UNALLOCATED, cudf::test::get_default_stream());
  thrust::sequence(rmm::exec_policy(cudf::test::get_default_stream()),
                   indices->mutable_view().begin<size_type>(),
                   indices->mutable_view().end<size_type>(),
                   0);
  return indices;
}

// generate the rows indices that should be checked for the child column of a list column.
//
// - if we are just checking for equivalence, we can skip any rows that are nulls. this allows
//   things like non-empty rows that have been nullified after creation.  they may actually contain
//   values, but since the row is null they don't matter for equivalency.
//
// - if we are checking for exact equality, we need to check all rows.
//
//   This allows us to differentiate between:
//
//  List<int32_t>:
//    Length : 1
//    Offsets : 0, 4
//    Null count: 1
//    0
//       0, 1, 2, 3
//
//  List<int32_t>:
//    Length : 1
//    Offsets : 0, 0
//    Null count: 1
//    0
//
std::unique_ptr<column> generate_child_row_indices(lists_column_view const& c,
                                                   column_view const& row_indices,
                                                   bool check_exact_equality)
{
  // if we are checking for exact equality, we should be checking for "unsanitized" data that may
  // be hiding underneath nulls. so check all rows instead of just non-null rows
  if (check_exact_equality) {
    return generate_all_row_indices(c.get_sliced_child(cudf::test::get_default_stream()).size());
  }

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
    cuda::proclaim_return_type<size_type>([row_indices = row_indices.begin<size_type>(),
                                           validity    = c.null_mask(),
                                           offsets     = c.offsets().begin<size_type>(),
                                           offset      = c.offset()] __device__(int index) {
      // both null mask and offsets data are not pre-sliced. so we need to add the column offset to
      // every incoming index.
      auto const true_index = row_indices[index] + offset;
      return !validity || cudf::bit_is_set(validity, true_index)
               ? (offsets[true_index + 1] - offsets[true_index])
               : 0;
    }));
  auto const output_size = thrust::reduce(rmm::exec_policy(cudf::test::get_default_stream()),
                                          row_size_iter,
                                          row_size_iter + row_indices.size());
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
  thrust::exclusive_scan(rmm::exec_policy(cudf::test::get_default_stream()),
                         row_size_iter,
                         row_size_iter + row_indices.size(),
                         output_row_start->mutable_view().begin<size_type>());

  // fill result column with 1s
  //
  // result = [1, 1, 1, 1, 1]
  //
  thrust::generate(rmm::exec_policy(cudf::test::get_default_stream()),
                   result->mutable_view().begin<size_type>(),
                   result->mutable_view().end<size_type>(),
                   cuda::proclaim_return_type<size_type>([] __device__() { return 1; }));

  // scatter the output row positions into result buffer
  //
  // result = [6, 1, 11, 1, 1]
  //
  auto output_row_iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [row_indices  = row_indices.begin<size_type>(),
       offsets      = c.offsets().begin<size_type>(),
       offset       = c.offset(),
       first_offset = cudf::detail::get_value<size_type>(
         c.offsets(), c.offset(), cudf::test::get_default_stream())] __device__(int index) {
        auto const true_index = row_indices[index] + offset;
        return offsets[true_index] - first_offset;
      }));
  thrust::scatter_if(rmm::exec_policy(cudf::test::get_default_stream()),
                     output_row_iter,
                     output_row_iter + row_indices.size(),
                     output_row_start->view().begin<size_type>(),
                     row_size_iter,
                     result->mutable_view().begin<size_type>(),
                     [] __device__(auto row_size) { return row_size != 0; });

  // generate keys for each output row
  //
  // result = [1, 1, 2, 2, 2]
  //
  auto keys =
    cudf::make_fixed_width_column(data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED);
  thrust::generate(rmm::exec_policy(cudf::test::get_default_stream()),
                   keys->mutable_view().begin<size_type>(),
                   keys->mutable_view().end<size_type>(),
                   cuda::proclaim_return_type<size_type>([] __device__() { return 0; }));
  thrust::scatter_if(rmm::exec_policy(cudf::test::get_default_stream()),
                     row_size_iter,
                     row_size_iter + row_indices.size(),
                     output_row_start->view().begin<size_type>(),
                     row_size_iter,
                     keys->mutable_view().begin<size_type>(),
                     [] __device__(auto row_size) { return row_size != 0; });
  thrust::inclusive_scan(rmm::exec_policy(cudf::test::get_default_stream()),
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
  thrust::inclusive_scan_by_key(rmm::exec_policy(cudf::test::get_default_stream()),
                                keys->view().begin<size_type>(),
                                keys->view().end<size_type>(),
                                result->view().begin<size_type>(),
                                result->mutable_view().begin<size_type>());
  return result;
}

#define PROP_EXPECT_EQ(a, b)                                \
  do {                                                      \
    if (verbosity == debug_output_level::QUIET) {           \
      if (a != b) { return false; }                         \
    } else {                                                \
      EXPECT_EQ(a, b);                                      \
      if (a != b) {                                         \
        if (verbosity == debug_output_level::FIRST_ERROR) { \
          return false;                                     \
        } else {                                            \
          result = false;                                   \
        }                                                   \
      }                                                     \
    }                                                       \
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
      cuda::proclaim_return_type<size_type>([row_indices = row_indices.begin<size_type>(),
                                             validity    = c.null_mask(),
                                             offset      = c.offset()] __device__(int index) {
        // both null mask and offsets data are not pre-sliced. so we need to add the column offset
        // to every incoming index.
        auto const true_index = row_indices[index] + offset;
        return !validity || cudf::bit_is_set(validity, true_index) ? 0 : 1;
      }));
    return thrust::reduce(rmm::exec_policy(cudf::test::get_default_stream()),
                          validity_iter,
                          validity_iter + row_indices.size());
  }

  bool compare_common(cudf::column_view const& lhs,
                      cudf::column_view const& rhs,
                      cudf::column_view const& lhs_row_indices,
                      cudf::column_view const& rhs_row_indices,
                      debug_output_level verbosity)
  {
    bool result = true;

    if (check_exact_equality) {
      PROP_EXPECT_EQ(lhs.type(), rhs.type());
    } else {
      PROP_EXPECT_EQ(types_equivalent(lhs.type(), rhs.type()), true);
    }

    auto const lhs_size = check_exact_equality ? lhs.size() : lhs_row_indices.size();
    auto const rhs_size = check_exact_equality ? rhs.size() : rhs_row_indices.size();
    PROP_EXPECT_EQ(lhs_size, rhs_size);

    if (lhs_size > 0 && check_exact_equality) { PROP_EXPECT_EQ(lhs.nullable(), rhs.nullable()); }

    // DISCUSSION: does this make sense, semantically?
    auto const lhs_null_count =
      check_exact_equality ? lhs.null_count() : count_nulls(lhs, lhs_row_indices);
    auto const rhs_null_count =
      check_exact_equality ? rhs.null_count() : count_nulls(rhs, rhs_row_indices);
    PROP_EXPECT_EQ(lhs_null_count, rhs_null_count);

    // equivalent, but not exactly equal columns can have a different number of children if their
    // sizes are both 0. Specifically, empty string columns may or may not have children.
    if (check_exact_equality || (lhs.size() > 0 && lhs.null_count() < lhs.size())) {
      PROP_EXPECT_EQ(lhs.num_children(), rhs.num_children());
    }

    return result;
  }

  template <typename T,
            std::enable_if_t<!std::is_same_v<T, cudf::list_view> &&
                             !std::is_same_v<T, cudf::struct_view>>* = nullptr>
  bool operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  debug_output_level verbosity)
  {
    return compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity);
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::list_view>>* = nullptr>
  bool operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  debug_output_level verbosity)
  {
    if (!compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity)) { return false; }

    cudf::lists_column_view lhs_l(lhs);
    cudf::lists_column_view rhs_l(rhs);

    // recurse

    // note: if a column is all nulls (and we are checking for exact equality) or otherwise empty,
    // no indices are generated and no recursion happens
    auto lhs_child_indices =
      generate_child_row_indices(lhs_l, lhs_row_indices, check_exact_equality);
    if (lhs_child_indices->size() > 0) {
      auto lhs_child = lhs_l.get_sliced_child(cudf::test::get_default_stream());
      auto rhs_child = rhs_l.get_sliced_child(cudf::test::get_default_stream());
      auto rhs_child_indices =
        generate_child_row_indices(rhs_l, rhs_row_indices, check_exact_equality);
      return cudf::type_dispatcher(lhs_child.type(),
                                   column_property_comparator<check_exact_equality>{},
                                   lhs_child,
                                   rhs_child,
                                   *lhs_child_indices,
                                   *rhs_child_indices,
                                   verbosity);
    }
    return true;
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::struct_view>>* = nullptr>
  bool operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  debug_output_level verbosity)
  {
    if (!compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity)) { return false; }

    structs_column_view l_scv(lhs);
    structs_column_view r_scv(rhs);

    for (size_type i = 0; i < lhs.num_children(); i++) {
      column_view lhs_child = l_scv.get_sliced_child(i, cudf::test::get_default_stream());
      column_view rhs_child = r_scv.get_sliced_child(i, cudf::test::get_default_stream());
      if (!cudf::type_dispatcher(lhs_child.type(),
                                 column_property_comparator<check_exact_equality>{},
                                 lhs_child,
                                 rhs_child,
                                 lhs_row_indices,
                                 rhs_row_indices,
                                 verbosity)) {
        return false;
      }
    }

    return true;
  }
};

template <typename DeviceComparator>
class corresponding_rows_unequal {
 public:
  corresponding_rows_unequal(column_device_view lhs_row_indices_,
                             column_device_view rhs_row_indices_,
                             size_type /*fp_ulps*/,
                             DeviceComparator comp_,
                             column_device_view /*lhs*/,
                             column_device_view /*rhs*/)
    : lhs_row_indices(lhs_row_indices_), rhs_row_indices(rhs_row_indices_), comp(comp_)
  {
  }

  __device__ bool operator()(size_type index)
  {
    using cudf::experimental::row::lhs_index_type;
    using cudf::experimental::row::rhs_index_type;

    return !comp(lhs_index_type{lhs_row_indices.element<size_type>(index)},
                 rhs_index_type{rhs_row_indices.element<size_type>(index)});
  }

  column_device_view lhs_row_indices;
  column_device_view rhs_row_indices;
  DeviceComparator comp;
};

template <typename DeviceComparator>
class corresponding_rows_not_equivalent {
  column_device_view lhs_row_indices;
  column_device_view rhs_row_indices;
  size_type const fp_ulps;
  DeviceComparator comp;
  column_device_view lhs;
  column_device_view rhs;

 public:
  corresponding_rows_not_equivalent(column_device_view lhs_row_indices_,
                                    column_device_view rhs_row_indices_,
                                    size_type fp_ulps_,
                                    DeviceComparator comp_,
                                    column_device_view lhs_,
                                    column_device_view rhs_)
    : lhs_row_indices(lhs_row_indices_),
      rhs_row_indices(rhs_row_indices_),
      fp_ulps(fp_ulps_),
      comp(comp_),
      lhs(lhs_),
      rhs(rhs_)
  {
  }

  struct typed_element_not_equivalent {
    template <typename T>
    __device__ std::enable_if_t<std::is_floating_point_v<T>, bool> operator()(
      column_device_view const& lhs,
      column_device_view const& rhs,
      size_type lhs_index,
      size_type rhs_index,
      size_type fp_ulps)
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
          T const abs_x_minus_y = std::abs(x - y);
          return abs_x_minus_y >= std::numeric_limits<T>::min() &&
                 abs_x_minus_y > std::numeric_limits<T>::epsilon() * std::abs(x + y) * fp_ulps;
        }
      } else {
        // if either is null, then the inequality was checked already
        return true;
      }
    }

    template <typename T, typename... Args>
    __device__ std::enable_if_t<not std::is_floating_point_v<T>, bool> operator()(Args...)
    {
      // Non-floating point inequality is checked already
      return true;
    }
  };

  __device__ bool operator()(size_type index)
  {
    using cudf::experimental::row::lhs_index_type;
    using cudf::experimental::row::rhs_index_type;

    auto const lhs_index = lhs_row_indices.element<size_type>(index);
    auto const rhs_index = rhs_row_indices.element<size_type>(index);

    if (not comp(lhs_index_type{lhs_index}, rhs_index_type{rhs_index})) {
      return type_dispatcher(
        lhs.type(), typed_element_not_equivalent{}, lhs, rhs, lhs_index, rhs_index, fp_ulps);
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
                                         debug_output_level verbosity,
                                         int depth)
{
  CUDF_EXPECTS(not differences.empty(), "Shouldn't enter this function if `differences` is empty");
  std::string const depth_str = depth > 0 ? "depth " + std::to_string(depth) + '\n' : "";
  // move the differences to the host.
  auto h_differences =
    cudf::detail::make_host_vector_sync(differences, cudf::test::get_default_stream());
  if (verbosity == debug_output_level::ALL_ERRORS) {
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
      cudf::detail::get_value<size_type>(lhs_row_indices, index, cudf::test::get_default_stream());
    auto const rhs_index =
      cudf::detail::get_value<size_type>(rhs_row_indices, index, cudf::test::get_default_stream());
    auto diff_lhs = cudf::slice(lhs, {lhs_index, lhs_index + 1}).front();
    auto diff_rhs = cudf::slice(rhs, {rhs_index, rhs_index + 1}).front();
    return depth_str + "first difference: " + "lhs[" + std::to_string(index) +
           "] = " + to_string(diff_lhs, "") + ", rhs[" + std::to_string(index) +
           "] = " + to_string(diff_rhs, "");
  }
}

// non-nested column types
template <typename T, bool check_exact_equality>
struct column_comparator_impl {
  bool operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  size_type fp_ulps,
                  int depth)
  {
    auto d_lhs_row_indices =
      cudf::column_device_view::create(lhs_row_indices, cudf::test::get_default_stream());
    auto d_rhs_row_indices =
      cudf::column_device_view::create(rhs_row_indices, cudf::test::get_default_stream());

    auto d_lhs = cudf::column_device_view::create(lhs, cudf::test::get_default_stream());
    auto d_rhs = cudf::column_device_view::create(rhs, cudf::test::get_default_stream());

    auto lhs_tview = table_view{{lhs}};
    auto rhs_tview = table_view{{rhs}};

    auto const comparator = cudf::experimental::row::equality::two_table_comparator{
      lhs_tview, rhs_tview, cudf::test::get_default_stream()};
    auto const has_nulls = cudf::has_nulls(lhs_tview) or cudf::has_nulls(rhs_tview);

    auto const device_comparator = comparator.equal_to<false>(cudf::nullate::DYNAMIC{has_nulls});

    using ComparatorType =
      std::conditional_t<check_exact_equality,
                         corresponding_rows_unequal<decltype(device_comparator)>,
                         corresponding_rows_not_equivalent<decltype(device_comparator)>>;

    auto differences = rmm::device_uvector<int>(
      lhs_row_indices.size(),
      cudf::test::get_default_stream());  // worst case: everything different
    auto input_iter = thrust::make_counting_iterator(0);

    auto diff_map =
      rmm::device_uvector<bool>(lhs_row_indices.size(), cudf::test::get_default_stream());

    thrust::transform(
      rmm::exec_policy(cudf::test::get_default_stream()),
      input_iter,
      input_iter + lhs_row_indices.size(),
      diff_map.begin(),
      ComparatorType(
        *d_lhs_row_indices, *d_rhs_row_indices, fp_ulps, device_comparator, *d_lhs, *d_rhs));

    auto diff_iter = thrust::copy_if(rmm::exec_policy(cudf::test::get_default_stream()),
                                     input_iter,
                                     input_iter + lhs_row_indices.size(),
                                     diff_map.begin(),
                                     differences.begin(),
                                     thrust::identity<bool>{});

    differences.resize(thrust::distance(differences.begin(), diff_iter),
                       cudf::test::get_default_stream());  // shrink back down

    if (not differences.is_empty()) {
      if (verbosity != debug_output_level::QUIET) {
        // GTEST_FAIL() does a return that conflicts with our return type. so hide it in a lambda.
        [&]() {
          GTEST_FAIL() << stringify_column_differences(
            differences, lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, depth);
        }();
      }
      return false;
    }
    return true;
  }
};

// forward declaration for nested-type recursion.
template <bool check_exact_equality>
struct column_comparator;

// specialization for list columns
template <bool check_exact_equality>
struct column_comparator_impl<list_view, check_exact_equality> {
  bool operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  size_type fp_ulps,
                  int depth)
  {
    lists_column_view lhs_l(lhs);
    lists_column_view rhs_l(rhs);

    CUDF_EXPECTS(lhs_row_indices.size() == rhs_row_indices.size(), "List column size mismatch");
    if (lhs_row_indices.is_empty()) { return true; }

    // worst case - everything is different
    rmm::device_uvector<int> differences(lhs_row_indices.size(), cudf::test::get_default_stream());

    // compare offsets, taking slicing into account

    // left side
    size_type lhs_shift = cudf::detail::get_value<size_type>(
      lhs_l.offsets(), lhs_l.offset(), cudf::test::get_default_stream());
    auto lhs_offsets = thrust::make_transform_iterator(
      lhs_l.offsets().begin<size_type>() + lhs_l.offset(),
      cuda::proclaim_return_type<size_type>(
        [lhs_shift] __device__(size_type offset) { return offset - lhs_shift; }));
    auto lhs_valids = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      cuda::proclaim_return_type<bool>(
        [mask = lhs_l.null_mask(), offset = lhs_l.offset()] __device__(size_type index) {
          return mask == nullptr ? true : cudf::bit_is_set(mask, index + offset);
        }));

    // right side
    size_type rhs_shift = cudf::detail::get_value<size_type>(
      rhs_l.offsets(), rhs_l.offset(), cudf::test::get_default_stream());
    auto rhs_offsets = thrust::make_transform_iterator(
      rhs_l.offsets().begin<size_type>() + rhs_l.offset(),
      cuda::proclaim_return_type<size_type>(
        [rhs_shift] __device__(size_type offset) { return offset - rhs_shift; }));
    auto rhs_valids = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      cuda::proclaim_return_type<bool>(
        [mask = rhs_l.null_mask(), offset = rhs_l.offset()] __device__(size_type index) {
          return mask == nullptr ? true : cudf::bit_is_set(mask, index + offset);
        }));

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
      rmm::exec_policy(cudf::test::get_default_stream()),
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
                       cudf::test::get_default_stream());  // shrink back down

    if (not differences.is_empty()) {
      if (verbosity != debug_output_level::QUIET) {
        // GTEST_FAIL() does a return that conflicts with our return type. so hide it in a lambda.
        [&]() {
          GTEST_FAIL() << stringify_column_differences(
            differences, lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, depth);
        }();
      }
      return false;
    }

    // recurse
    // note: if a column is all nulls (and we are only checking for equivalence) or otherwise empty,
    // no indices are generated and no recursion happens
    auto lhs_child_indices =
      generate_child_row_indices(lhs_l, lhs_row_indices, check_exact_equality);
    if (lhs_child_indices->size() > 0) {
      auto lhs_child = lhs_l.get_sliced_child(cudf::test::get_default_stream());
      auto rhs_child = rhs_l.get_sliced_child(cudf::test::get_default_stream());
      auto rhs_child_indices =
        generate_child_row_indices(rhs_l, rhs_row_indices, check_exact_equality);
      return cudf::type_dispatcher(lhs_child.type(),
                                   column_comparator<check_exact_equality>{},
                                   lhs_child,
                                   rhs_child,
                                   *lhs_child_indices,
                                   *rhs_child_indices,
                                   verbosity,
                                   fp_ulps,
                                   depth + 1);
    }

    return true;
  }
};

template <bool check_exact_equality>
struct column_comparator_impl<struct_view, check_exact_equality> {
  bool operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  size_type fp_ulps,
                  int depth)
  {
    structs_column_view l_scv(lhs);
    structs_column_view r_scv(rhs);

    for (size_type i = 0; i < lhs.num_children(); i++) {
      column_view lhs_child = l_scv.get_sliced_child(i, cudf::test::get_default_stream());
      column_view rhs_child = r_scv.get_sliced_child(i, cudf::test::get_default_stream());
      if (!cudf::type_dispatcher(lhs_child.type(),
                                 column_comparator<check_exact_equality>{},
                                 lhs_child,
                                 rhs_child,
                                 lhs_row_indices,
                                 rhs_row_indices,
                                 verbosity,
                                 fp_ulps,
                                 depth + 1)) {
        return false;
      }
    }
    return true;
  }
};

template <bool check_exact_equality>
struct column_comparator {
  template <typename T>
  bool operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  size_type fp_ulps,
                  int depth = 0)
  {
    // compare properties
    if (!cudf::type_dispatcher(lhs.type(),
                               column_property_comparator<check_exact_equality>{},
                               lhs,
                               rhs,
                               lhs_row_indices,
                               rhs_row_indices,
                               verbosity)) {
      return false;
    }

    // compare values
    column_comparator_impl<T, check_exact_equality> comparator{};
    return comparator(lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, fp_ulps, depth);
  }
};

}  // namespace

namespace detail {
/**
 * @copydoc cudf::test::expect_column_properties_equal
 */
bool expect_column_properties_equal(column_view const& lhs,
                                    column_view const& rhs,
                                    debug_output_level verbosity)
{
  auto lhs_indices = generate_all_row_indices(lhs.size());
  auto rhs_indices = generate_all_row_indices(rhs.size());
  return cudf::type_dispatcher(lhs.type(),
                               column_property_comparator<true>{},
                               lhs,
                               rhs,
                               *lhs_indices,
                               *rhs_indices,
                               verbosity);
}

/**
 * @copydoc cudf::test::expect_column_properties_equivalent
 */
bool expect_column_properties_equivalent(column_view const& lhs,
                                         column_view const& rhs,
                                         debug_output_level verbosity)
{
  auto lhs_indices = generate_all_row_indices(lhs.size());
  auto rhs_indices = generate_all_row_indices(rhs.size());
  return cudf::type_dispatcher(lhs.type(),
                               column_property_comparator<false>{},
                               lhs,
                               rhs,
                               *lhs_indices,
                               *rhs_indices,
                               verbosity);
}

/**
 * @copydoc cudf::test::expect_columns_equal
 */
bool expect_columns_equal(cudf::column_view const& lhs,
                          cudf::column_view const& rhs,
                          debug_output_level verbosity)
{
  auto lhs_indices = generate_all_row_indices(lhs.size());
  auto rhs_indices = generate_all_row_indices(rhs.size());
  return cudf::type_dispatcher(lhs.type(),
                               column_comparator<true>{},
                               lhs,
                               rhs,
                               *lhs_indices,
                               *rhs_indices,
                               verbosity,
                               cudf::test::default_ulp);
}

/**
 * @copydoc cudf::test::expect_columns_equivalent
 */
bool expect_columns_equivalent(cudf::column_view const& lhs,
                               cudf::column_view const& rhs,
                               debug_output_level verbosity,
                               size_type fp_ulps)
{
  auto lhs_indices = generate_all_row_indices(lhs.size());
  auto rhs_indices = generate_all_row_indices(rhs.size());
  return cudf::type_dispatcher(lhs.type(),
                               column_comparator<false>{},
                               lhs,
                               rhs,
                               *lhs_indices,
                               *rhs_indices,
                               verbosity,
                               fp_ulps);
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
  EXPECT_TRUE(thrust::equal(rmm::exec_policy(cudf::test::get_default_stream()),
                            typed_lhs,
                            typed_lhs + size_bytes,
                            typed_rhs));
}
}  // namespace detail

/**
 * @copydoc cudf::test::expect_column_empty
 */
void expect_column_empty(cudf::column_view const& col)
{
  EXPECT_EQ(0, col.size());
  EXPECT_EQ(0, col.null_count());
}

/**
 * @copydoc cudf::test::bitmask_to_host
 */
std::vector<bitmask_type> bitmask_to_host(cudf::column_view const& c)
{
  if (c.nullable()) {
    auto num_bitmasks = num_bitmask_words(c.size());
    std::vector<bitmask_type> host_bitmask(num_bitmasks);
    if (c.offset() == 0) {
      CUDF_CUDA_TRY(cudaMemcpy(host_bitmask.data(),
                               c.null_mask(),
                               num_bitmasks * sizeof(bitmask_type),
                               cudaMemcpyDefault));
    } else {
      auto mask = copy_bitmask(c.null_mask(), c.offset(), c.offset() + c.size());
      CUDF_CUDA_TRY(cudaMemcpy(
        host_bitmask.data(), mask.data(), num_bitmasks * sizeof(bitmask_type), cudaMemcpyDefault));
    }

    return host_bitmask;
  } else {
    return std::vector<bitmask_type>{};
  }
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
