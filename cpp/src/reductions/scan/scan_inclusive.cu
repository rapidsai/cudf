/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "scan.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/table/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <structs/utilities.hpp>

#include <thrust/scan.h>

namespace cudf {
namespace detail {

// logical-and scan of the null mask of the input view
rmm::device_buffer mask_scan(column_view const& input_view,
                             scan_type inclusive,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  rmm::device_buffer mask =
    detail::create_null_mask(input_view.size(), mask_state::UNINITIALIZED, stream, mr);
  auto d_input   = column_device_view::create(input_view, stream);
  auto valid_itr = detail::make_validity_iterator(*d_input);

  auto first_null_position = [&] {
    size_type const first_null = thrust::find_if_not(rmm::exec_policy(stream),
                                                     valid_itr,
                                                     valid_itr + input_view.size(),
                                                     thrust::identity<bool>{}) -
                                 valid_itr;
    size_type const exclusive_offset = (inclusive == scan_type::EXCLUSIVE) ? 1 : 0;
    return std::min(input_view.size(), first_null + exclusive_offset);
  }();

  set_null_mask(static_cast<bitmask_type*>(mask.data()), 0, first_null_position, true, stream);
  set_null_mask(
    static_cast<bitmask_type*>(mask.data()), first_null_position, input_view.size(), false, stream);
  return mask;
}

namespace {

/**
 * @brief Strings inclusive scan operator
 *
 * This was specifically created to workaround a thrust issue
 * https://github.com/NVIDIA/thrust/issues/1479
 * where invalid values are passed to the operator.
 *
 * This operator will accept index values, check them and then
 * run the `Op` operation on the individual string_view objects.
 * The returned result is the appropriate index value.
 */
template <typename Op>
struct string_scan_operator {
  column_device_view const col;          ///< strings column device view
  string_view const null_replacement{};  ///< value used when element is null
  bool const has_nulls;                  ///< true if col has null elements

  string_scan_operator(column_device_view const& col, bool has_nulls = true)
    : col{col}, null_replacement{Op::template identity<string_view>()}, has_nulls{has_nulls}
  {
    CUDF_EXPECTS(type_id::STRING == col.type().id(), "the data type mismatch");
    // verify validity bitmask is non-null, otherwise, is_null_nocheck() will crash
    if (has_nulls) CUDF_EXPECTS(col.nullable(), "column with nulls must have a validity bitmask");
  }

  CUDA_DEVICE_CALLABLE
  size_type operator()(size_type lhs, size_type rhs) const
  {
    // thrust::inclusive_scan may pass us garbage values so we need to protect ourselves;
    // in these cases the return value does not matter since the result is not used
    if (lhs < 0 || rhs < 0 || lhs >= col.size() || rhs >= col.size()) return 0;
    string_view d_lhs =
      has_nulls && col.is_null_nocheck(lhs) ? null_replacement : col.element<string_view>(lhs);
    string_view d_rhs =
      has_nulls && col.is_null_nocheck(rhs) ? null_replacement : col.element<string_view>(rhs);
    return Op{}(d_lhs, d_rhs) == d_lhs ? lhs : rhs;
  }
};

/**
 * @brief Dispatcher for running a Scan operation on an input column
 *
 * @tparam Op device binary operator
 */
template <typename Op>
struct scan_dispatcher {
 private:
  template <typename T>
  static constexpr bool is_string_supported()
  {
    return std::is_same_v<T, string_view> &&
           (std::is_same_v<Op, DeviceMin> || std::is_same_v<Op, DeviceMax>);
  }

  template <typename T>
  static constexpr bool is_supported()
  {
    return cuda::std::is_arithmetic<T>() || is_string_supported<T>();
  }

  // for arithmetic types
  template <typename T, std::enable_if_t<cuda::std::is_arithmetic<T>::value>* = nullptr>
  auto inclusive_scan(column_view const& input_view,
                      null_policy,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
  {
    auto output_column = detail::allocate_like(
      input_view, input_view.size(), mask_allocation_policy::NEVER, stream, mr);
    mutable_column_view result = output_column->mutable_view();

    auto d_input = column_device_view::create(input_view, stream);
    auto const begin =
      make_null_replacement_iterator(*d_input, Op::template identity<T>(), input_view.has_nulls());
    thrust::inclusive_scan(
      rmm::exec_policy(stream), begin, begin + input_view.size(), result.data<T>(), Op{});

    CHECK_CUDA(stream.value());
    return output_column;
  }

  // for string type: only MIN and MAX are supported
  template <typename T, std::enable_if_t<is_string_supported<T>()>* = nullptr>
  std::unique_ptr<column> inclusive_scan(column_view const& input_view,
                                         null_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
  {
    auto d_input = column_device_view::create(input_view, stream);

    // build indices of the scan operation results
    rmm::device_uvector<size_type> result(input_view.size(), stream);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           thrust::counting_iterator<size_type>(0),
                           thrust::counting_iterator<size_type>(input_view.size()),
                           result.begin(),
                           string_scan_operator<Op>{*d_input, input_view.has_nulls()});

    // call gather using the indices to build the output column
    return cudf::strings::detail::gather(
      strings_column_view(input_view), result.begin(), result.end(), false, stream, mr);
  }

 public:
  /**
   * @brief Creates a new column from the input column by applying the scan operation
   *
   * @param input Input column view
   * @param null_handling How null row entries are to be processed
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return
   *
   * @tparam T type of input column
   */
  template <typename T, typename std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return inclusive_scan<T>(input, null_handling, stream, mr);
  }

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("Non-arithmetic types not supported for inclusive scan");
  }
};

/**
 * @brief generate row ranks or dense ranks using a row comparison then scan the results
 *
 * @tparam has_nulls if the order_by column has nulls
 * @tparam value_resolver flag value resolver with boolean first and row number arguments
 * @tparam scan_operator scan function ran on the flag values
 * @param order_by input column to generate ranks for
 * @param resolver flag value resolver
 * @param scan_op scan operation ran on the flag results
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> rank values
 */
template <bool has_nulls, typename value_resolver, typename scan_operator>
std::unique_ptr<column> rank_generator(column_view const& order_by,
                                       value_resolver resolver,
                                       scan_operator scan_op,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const superimposed = structs::detail::superimpose_parent_nulls(order_by, stream, mr);
  table_view const order_table{{std::get<0>(superimposed)}};
  auto const flattener = cudf::structs::detail::flatten_nested_columns(
    order_table, {}, {}, structs::detail::column_nullability::MATCH_INCOMING);
  auto const d_flat_order = table_device_view::create(std::get<0>(flattener), stream);
  row_equality_comparator<has_nulls> comparator(*d_flat_order, *d_flat_order, true);
  auto ranks         = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                       order_table.num_rows(),
                                       mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  auto mutable_ranks = ranks->mutable_view();

  thrust::tabulate(rmm::exec_policy(stream),
                   mutable_ranks.begin<size_type>(),
                   mutable_ranks.end<size_type>(),
                   [comparator, resolver] __device__(size_type row_index) {
                     return resolver(row_index == 0 || !comparator(row_index, row_index - 1),
                                     row_index);
                   });

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         mutable_ranks.begin<size_type>(),
                         mutable_ranks.end<size_type>(),
                         mutable_ranks.begin<size_type>(),
                         scan_op);
  return ranks;
}

}  // namespace

std::unique_ptr<column> inclusive_dense_rank_scan(column_view const& order_by,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in dense_rank scan.");
  if (has_nested_nulls(table_view{{order_by}})) {
    return rank_generator<true>(
      order_by,
      [] __device__(bool equality, auto row_index) { return equality; },
      DeviceSum{},
      stream,
      mr);
  }
  return rank_generator<false>(
    order_by,
    [] __device__(bool equality, auto row_index) { return equality; },
    DeviceSum{},
    stream,
    mr);
}

std::unique_ptr<column> inclusive_rank_scan(column_view const& order_by,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in rank scan.");
  if (has_nested_nulls(table_view{{order_by}})) {
    return rank_generator<true>(
      order_by,
      [] __device__(bool equality, auto row_index) { return equality ? row_index + 1 : 0; },
      DeviceMax{},
      stream,
      mr);
  }
  return rank_generator<false>(
    order_by,
    [] __device__(bool equality, auto row_index) { return equality ? row_index + 1 : 0; },
    DeviceMax{},
    stream,
    mr);
}

std::unique_ptr<column> scan_inclusive(
  column_view const& input,
  std::unique_ptr<aggregation> const& agg,
  null_policy null_handling,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto output = scan_agg_dispatch<scan_dispatcher>(input, agg, null_handling, stream, mr);

  if (agg->kind == aggregation::RANK || agg->kind == aggregation::DENSE_RANK) {
    return output;
  } else if (null_handling == null_policy::EXCLUDE) {
    output->set_null_mask(detail::copy_bitmask(input, stream, mr), input.null_count());
  } else if (input.nullable()) {
    output->set_null_mask(mask_scan(input, scan_type::INCLUSIVE, stream, mr), UNKNOWN_NULL_COUNT);
  }

  return output;
}
}  // namespace detail
}  // namespace cudf
