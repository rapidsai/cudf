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
    rmm::device_uvector<size_type> result(input_view.size(), stream, mr);
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

/**
 * @brief Solve a recurrence relation using a blelloch scan
 * The second elements of the pairs will contain the result
 */
void compute_recurrence(rmm::device_uvector<thrust::pair<double, double>>& input,
                        rmm::cuda_stream_view stream)
{
  blelloch_functor op;
  thrust::inclusive_scan(rmm::exec_policy(stream), input.begin(), input.end(), input.begin(), op);
}

/**
 * @brief modify the source pairs that eventually yield the numerator
 * and denoninator to account for nan values. Pairs at nan indicies
 * become the identity operator (1, 0). The first pair after a nan
 * value or sequence of nan values has its first element multiplied by
 * N factors of beta, where N is the number of preceeding NaNs.
 */
void pair_beta_adjust(column_view const& input,
                      rmm::device_uvector<thrust::pair<double, double>>& pairs,
                      double beta,
                      rmm::cuda_stream_view stream)
{
  auto device_view = *column_device_view::create(input);
  auto valid_it    = cudf::detail::make_validity_iterator(device_view);

  // Holds count of nulls
  rmm::device_vector<int> nullcnt(input.size());

  // TODO - not sure why two iterators produce a different result
  // Invert the null iterator
  thrust::transform(rmm::exec_policy(stream),
                    valid_it,
                    valid_it + input.size(),
                    nullcnt.begin(),
                    [=] __host__ __device__(bool valid) -> bool { return 1 - valid; });

  // 0, 1, 0, 1, 1, 0 -> 0, 0, 1, 0, 0, 2
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                nullcnt.begin(),
                                nullcnt.end() - 1,
                                nullcnt.begin(),
                                nullcnt.begin() + 1);

  valid_it = cudf::detail::make_validity_iterator(device_view);
  thrust::transform(
    rmm::exec_policy(stream),
    valid_it,
    valid_it + input.size(),
    pairs.begin(),
    pairs.begin(),
    [=] __host__ __device__(bool valid,
                            thrust::pair<double, double> pair) -> thrust::pair<double, double> {
      if (!valid) {
        return thrust::pair<double, double>(1.0, 0.0);
      } else {
        return pair;
      }
    });

  valid_it           = cudf::detail::make_validity_iterator(device_view);
  auto valid_and_exp = thrust::make_zip_iterator(thrust::make_tuple(valid_it, nullcnt.begin()));

  thrust::transform(
    rmm::exec_policy(stream),
    valid_and_exp,
    valid_and_exp + input.size(),
    pairs.begin(),
    pairs.begin(),
    [=] __host__ __device__(thrust::tuple<bool, int> valid_and_exp,
                            thrust::pair<double, double> pair) -> thrust::pair<double, double> {
      bool valid = thrust::get<0>(valid_and_exp);
      int exp    = thrust::get<1>(valid_and_exp);
      if (valid & (exp != 0)) {
        double beta  = thrust::get<0>(pair);
        double scale = thrust::get<1>(pair);
        return thrust::pair<double, double>(beta * (pow(beta, exp)), scale);
      } else {
        return pair;
      }
    });
}

rmm::device_uvector<double> compute_ewma_adjust(column_view const& input,
                                                double beta,
                                                rmm::cuda_stream_view stream,
                                                rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<double> output(input.size(), stream, mr);
  rmm::device_uvector<thrust::pair<double, double>> pairs(input.size(), stream, mr);

  // Numerator
  // Fill with pairs
  thrust::transform(rmm::exec_policy(stream),
                    input.begin<double>(),
                    input.end<double>(),
                    pairs.begin(),
                    [=] __host__ __device__(double input) -> thrust::pair<double, double> {
                      return thrust::pair<double, double>(beta, input);
                    });

  if (input.has_nulls()) { pair_beta_adjust(input, pairs, beta, stream); }

  compute_recurrence(pairs, stream);

  // copy the second elements to the output for now
  thrust::transform(rmm::exec_policy(stream),
                    pairs.begin(),
                    pairs.end(),
                    output.begin(),
                    [=] __host__ __device__(thrust::pair<double, double> pair) -> double {
                      return thrust::get<1>(pair);
                    });

  // Denominator
  // Fill with pairs
  thrust::fill(
    rmm::exec_policy(stream), pairs.begin(), pairs.end(), thrust::pair<double, double>(beta, 1.0));

  if (input.has_nulls()) { pair_beta_adjust(input, pairs, beta, stream); }
  compute_recurrence(pairs, stream);

  thrust::transform(
    rmm::exec_policy(stream),
    pairs.begin(),
    pairs.end(),
    output.begin(),
    output.begin(),
    [=] __host__ __device__(thrust::pair<double, double> pair, double numerator) -> double {
      return numerator / thrust::get<1>(pair);
    });
  return output;
}

rmm::device_uvector<double> compute_ewma_noadjust(column_view const& input,
                                                  double beta,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<double> output(input.size(), stream, mr);
  rmm::device_uvector<thrust::pair<double, double>> pairs(input.size(), stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    input.begin<double>(),
                    input.end<double>(),
                    pairs.begin(),
                    [=] __host__ __device__(double input) -> thrust::pair<double, double> {
                      return thrust::pair<double, double>(beta, (1.0 - beta) * input);
                    });

  // TODO: the first pair is WRONG using the above. Reset just that pair

  thrust::transform(rmm::exec_policy(stream),
                    input.begin<double>(),
                    input.begin<double>() + 1,
                    pairs.begin(),
                    [=] __host__ __device__(double input) -> thrust::pair<double, double> {
                      return thrust::pair<double, double>(beta, input);
                    });
  if (input.has_nulls()) {
    auto device_view = *column_device_view::create(input);
    auto valid_it    = cudf::detail::make_validity_iterator(device_view);

    thrust::transform(
      rmm::exec_policy(stream),
      valid_it,
      valid_it + input.size(),
      pairs.begin(),
      pairs.begin(),
      [=] __host__ __device__(bool valid,
                              thrust::pair<double, double> pair) -> thrust::pair<double, double> {
        if (!valid) {
          return thrust::pair<double, double>(1.0, 0.0);
        } else {
          return pair;
        }
      });
  }
  compute_recurrence(pairs, stream);
  // copy the second elements to the output for now
  thrust::transform(rmm::exec_policy(stream),
                    pairs.begin(),
                    pairs.end(),
                    output.begin(),
                    [=] __host__ __device__(thrust::pair<double, double> pair) -> double {
                      return thrust::get<1>(pair);
                    });
  return output;
}

std::unique_ptr<column> ewma(column_view const& input,
                             double com,
                             bool adjust,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_floating_point(input.type()), "Column must be floating point type");

  double beta = 1.0 - (1.0 / (com + 1.0));

  rmm::device_uvector<double> data(input.size(), stream, mr);
  if (adjust) {
    data = compute_ewma_adjust(input, beta, stream, mr);
  } else {
    data = compute_ewma_noadjust(input, beta, stream, mr);
  }
  auto col = std::make_unique<column>(
    cudf::data_type{cudf::type_id::FLOAT64}, input.size(), std::move(data.release()));
  return col;
}

void print_col(std::unique_ptr<column>& input, rmm::cuda_stream_view stream)
{
  auto real_input = input.get()[0].view();
  rmm::device_vector<double> input_vec(input.get()[0].size());
  thrust::copy(rmm::exec_policy(stream),
               real_input.begin<double>(),
               real_input.end<double>(),
               input_vec.begin());
  thrust::host_vector<double> input_vec_host = input_vec;

  for (int i = 0; i < real_input.size(); i++) {
    std::cout << input_vec_host[i] << " ";
  }
  std::cout << std::endl;
}

/**
 * @brief Compute exponentially weighted moving variance
 * EWMVAR[i] is defined as EWMVAR[i] = EWMA[xi**2] - EWMA[xi]**2
 */
std::unique_ptr<column> ewmvar(column_view const& input,
                               double com,
                               bool adjust,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  // get xi**2
  std::unique_ptr<column> xi_sqr = make_fixed_width_column(
    cudf::data_type{cudf::type_id::FLOAT64}, input.size(), copy_bitmask(input));
  mutable_column_view xi_sqr_d = xi_sqr->mutable_view();
  thrust::transform(rmm::exec_policy(stream),
                    input.begin<double>(),
                    input.end<double>(),
                    xi_sqr_d.begin<double>(),
                    [=] __host__ __device__(double input) -> double { return input * input; });
  print_col(xi_sqr, stream);

  // get EWMA[xi**2]
  std::unique_ptr<column> ewma_xi_sqr = ewma((*xi_sqr).view(), com, adjust, stream, mr);
  print_col(ewma_xi_sqr, stream);

  // get EWMA[xi]
  std::unique_ptr<column> ewma_xi = ewma(input, com, adjust, stream, mr);
  print_col(ewma_xi, stream);

  // reuse the memory from computing xi_sqr to write the output
  thrust::transform(
    rmm::exec_policy(stream),
    ewma_xi.get()[0].view().begin<double>(),
    ewma_xi.get()[0].view().end<double>(),
    ewma_xi_sqr.get()[0].view().begin<double>(),
    ewma_xi.get()[0].mutable_view().begin<double>(),
    [=] __host__ __device__(double x, double xsqrd) -> double { return xsqrd - x * x; });

  // return means;
  return ewma_xi;
}

std::unique_ptr<column> ewmstd(column_view const& input,
                               double com,
                               bool adjust,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  std::unique_ptr<column> var = ewmvar(input, com, adjust, stream, mr);
  auto var_view               = var.get()[0].mutable_view();

  // write into the same memory
  thrust::transform(rmm::exec_policy(stream),
                    var_view.begin<double>(),
                    var_view.end<double>(),
                    var_view.begin<double>(),
                    [=] __host__ __device__(double input) -> double { return sqrt(input); });

  return var;
}

std::unique_ptr<column> ewm(column_view const& input,
                            std::unique_ptr<aggregation> const& agg,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr)
{
  switch (agg->kind) {
    case aggregation::EWMA: {
      double com  = (dynamic_cast<ewma_aggregation*>(agg.get()))->com;
      bool adjust = (dynamic_cast<ewma_aggregation*>(agg.get()))->adjust;
      return ewma(input, com, adjust, stream, mr);
    }
    case aggregation::EWMVAR: {
      double com  = (dynamic_cast<ewmvar_aggregation*>(agg.get()))->com;
      bool adjust = (dynamic_cast<ewmvar_aggregation*>(agg.get()))->adjust;
      return ewmvar(input, com, adjust, stream, mr);
    }
    case aggregation::EWMSTD: {
      double com  = (dynamic_cast<ewmstd_aggregation*>(agg.get()))->com;
      bool adjust = (dynamic_cast<ewmstd_aggregation*>(agg.get()))->adjust;
      return ewmstd(input, com, adjust, stream, mr);
    }
    default: CUDF_FAIL("Unsupported aggregation operator for scan");
  }
}

std::unique_ptr<column> scan_inclusive(
  column_view const& input,
  std::unique_ptr<aggregation> const& agg,
  null_policy null_handling,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto output = scan_agg_dispatch<scan_dispatcher>(input, agg, null_handling, stream, mr);

  if (agg->kind == aggregation::RANK || agg->kind == aggregation::DENSE_RANK ||
      agg->kind == aggregation::EWMA || agg->kind == aggregation::EWMVAR ||
      agg->kind == aggregation::EWMSTD) {
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
