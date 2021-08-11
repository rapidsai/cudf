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

#include <thrust/logical.h>
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
    return std::is_arithmetic<T>::value || is_string_supported<T>();
  }

  // for arithmetic types
  template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
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

template <bool has_nested_nulls>
std::unique_ptr<column> generate_dense_ranks(column_view const& order_by,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto const flat_order =
    order_by.type().id() == type_id::STRUCT
      ? table_view{std::vector<column_view>{order_by.child_begin(), order_by.child_end()}}
      : table_view{{order_by}};
  auto const d_flat_order = table_device_view::create(flat_order, stream);
  row_equality_comparator<has_nested_nulls> comparator(*d_flat_order, *d_flat_order, true);
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, order_by.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();

  if (order_by.type().id() == type_id::STRUCT && order_by.has_nulls()) {
    auto const d_col_order = column_device_view::create(order_by, stream);
    thrust::tabulate(rmm::exec_policy(stream),
                     mutable_ranks.begin<size_type>(),
                     mutable_ranks.end<size_type>(),
                     [comparator, d_col_order = *d_col_order] __device__(size_type row_index) {
                       if (row_index == 0) { return true; }
                       bool const lhs_is_null{d_col_order.is_null(row_index)};
                       bool const rhs_is_null{d_col_order.is_null(row_index - 1)};
                       if (lhs_is_null && rhs_is_null) {
                         return false;
                       } else if (lhs_is_null != rhs_is_null) {
                         return true;
                       }
                       return !comparator(row_index, row_index - 1);
                     });
  } else {
    thrust::tabulate(rmm::exec_policy(stream),
                     mutable_ranks.begin<size_type>(),
                     mutable_ranks.end<size_type>(),
                     [comparator] __device__(size_type row_index) {
                       return row_index == 0 || !comparator(row_index, row_index - 1);
                     });
  }

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         mutable_ranks.begin<size_type>(),
                         mutable_ranks.end<size_type>(),
                         mutable_ranks.begin<size_type>());
  return ranks;
}

template <bool has_nested_nulls>
std::unique_ptr<column> generate_ranks(column_view const& order_by,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const flat_order =
    order_by.type().id() == type_id::STRUCT
      ? table_view{std::vector<column_view>{order_by.child_begin(), order_by.child_end()}}
      : table_view{{order_by}};
  auto const d_flat_order = table_device_view::create(flat_order, stream);
  row_equality_comparator<has_nested_nulls> comparator(*d_flat_order, *d_flat_order, true);
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, order_by.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();

  if (order_by.type().id() == type_id::STRUCT && order_by.has_nulls()) {
    auto const d_col_order = column_device_view::create(order_by, stream);
    thrust::tabulate(rmm::exec_policy(stream),
                     mutable_ranks.begin<size_type>(),
                     mutable_ranks.end<size_type>(),
                     [comparator, d_col_order = *d_col_order] __device__(size_type row_index) {
                       if (row_index == 0) { return 1; }
                       bool const lhs_is_null{d_col_order.is_null(row_index)};
                       bool const rhs_is_null{d_col_order.is_null(row_index - 1)};
                       if (lhs_is_null and rhs_is_null) {
                         return 0;
                       } else if (lhs_is_null != rhs_is_null) {
                         return row_index + 1;
                       }
                       return comparator(row_index, row_index - 1) ? 0 : row_index + 1;
                     });
  } else {
    thrust::tabulate(
      rmm::exec_policy(stream),
      mutable_ranks.begin<size_type>(),
      mutable_ranks.end<size_type>(),
      [comparator] __device__(size_type row_index) {
        return row_index != 0 && comparator(row_index, row_index - 1) ? 0 : row_index + 1;
      });
  }

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         mutable_ranks.begin<size_type>(),
                         mutable_ranks.end<size_type>(),
                         mutable_ranks.begin<size_type>(),
                         DeviceMax{});
  return ranks;
}

}  // namespace

std::unique_ptr<column> inclusive_dense_rank_scan(column_view const& order_by,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(order_by.type().id() != type_id::LIST, "Unsupported list type in dense_rank scan.");
  CUDF_EXPECTS(std::none_of(order_by.child_begin(),
                            order_by.child_end(),
                            [](auto const& col) { return is_nested(col.type()); }),
               "Unsupported nested columns in dense_rank scan.");
  if ((order_by.type().id() == type_id::STRUCT &&
       has_nested_nulls(
         table_view{std::vector<column_view>{order_by.child_begin(), order_by.child_end()}})) ||
      (order_by.type().id() != type_id::STRUCT && order_by.has_nulls())) {
    return generate_dense_ranks<true>(order_by, stream, mr);
  }
  return generate_dense_ranks<false>(order_by, stream, mr);
}

std::unique_ptr<column> inclusive_rank_scan(column_view const& order_by,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(order_by.type().id() != type_id::LIST, "Unsupported list type in rank scan.");
  CUDF_EXPECTS(std::none_of(order_by.child_begin(),
                            order_by.child_end(),
                            [](auto const& col) { return is_nested(col.type()); }),
               "Unsupported nested columns in rank scan.");
  if ((order_by.type().id() == type_id::STRUCT &&
       has_nested_nulls(
         table_view{std::vector<column_view>{order_by.child_begin(), order_by.child_end()}})) ||
      (order_by.type().id() != type_id::STRUCT && order_by.has_nulls())) {
    return generate_ranks<true>(order_by, stream, mr);
  }
  return generate_ranks<false>(order_by, stream, mr);
}

class blelloch_functor {
public:
 __device__ thrust::pair<double, double> operator()(thrust::pair<double, double> ci,
                                                    thrust::pair<double, double> cj)
 {
   double ci0 = thrust::get<0>(ci);
   double ci1 = thrust::get<1>(ci);
   double cj0 = thrust::get<0>(cj);
   double cj1 = thrust::get<1>(cj);
   return thrust::pair<double, double>(ci0 * cj0, ci1 * cj0 + cj1);
 }
};

rmm::device_vector<double> compute_recurrence(rmm::device_vector<thrust::pair<double, double>> input) {
  // final result
  rmm::device_vector<double> result(input.size());
  
  blelloch_functor op;
  thrust::inclusive_scan(input.begin(), input.end(), input.begin(), op);
  thrust::transform(input.begin(),
                    input.end(),
                    result.begin(),
                    [=] __host__ __device__(thrust::pair<double, double> input) -> double {
                      return thrust::get<1>(input);
                    });
  return result;

}

rmm::device_vector<double> ewm_numerator(column_view const& input, double beta)
{
  rmm::device_vector<double> output(input.size());
  rmm::device_vector<thrust::pair<double, double>> pairs(input.size());
  thrust::transform(input.begin<double>(),
                    input.end<double>(),
                    pairs.begin(),
                    [=] __host__ __device__(double input) -> thrust::pair<double, double> {
                      return thrust::pair<double, double>(beta, input);
                    });

  rmm::device_vector<double> result = compute_recurrence(pairs);
  return result;
}

rmm::device_vector<double> ewm_denominator(column_view const& input, double beta)
{
  rmm::device_vector<double> output(input.size());
  rmm::device_vector<thrust::pair<double, double>> pairs(input.size());
  thrust::fill(pairs.begin(), pairs.end(), thrust::pair<double, double>(beta, 1.0));

  rmm::device_vector<double> result = compute_recurrence(pairs);
  return result;
}


std::unique_ptr<column> ewm(column_view const& input,
                            double com,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type() == cudf::data_type{cudf::type_id::FLOAT64},
               "Column must be float64 type");

  double beta = 1.0 - (1.0 / (com + 1.0));
    
  auto output = make_fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64}, input.size());
  auto output_mutable_view = output->mutable_view();

  auto begin = output_mutable_view.begin<double>();
  auto end   = output_mutable_view.end<double>();

  rmm::device_vector<double> denominator = ewm_denominator(input, beta);
  rmm::device_vector<double> numerator   = ewm_numerator(input, beta);

  thrust::transform(rmm::exec_policy(stream),
                    numerator.begin(),
                    numerator.end(),
                    denominator.begin(),
                    output_mutable_view.begin<double>(),
                    thrust::divides<double>());

  return output;
}


std::unique_ptr<column> ewma(
  column_view const& input, 
  rmm::cuda_stream_view stream, 
  rmm::mr::device_memory_resource* mr) 
{
  double com = 0.5;
  std::unique_ptr<column> result = ewm(input, com, stream, mr);
  return result;
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
