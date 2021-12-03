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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

namespace cudf {
namespace detail {

template <typename T>
class recurrence_functor {
  using pair_type = thrust::pair<T, T>;

 public:
  __device__ pair_type operator()(pair_type ci, pair_type cj)
  {
    return {ci.first * cj.first, ci.second * cj.first + cj.second};
  }
};

/**
 * @brief Solve a recurrence relation using a blelloch scan
 * The second elements of the pairs will contain the result
 */
template <typename T>
void compute_recurrence(rmm::device_uvector<thrust::pair<T, T>>& input,
                        rmm::cuda_stream_view stream)
{
  thrust::inclusive_scan(rmm::exec_policy(stream), input.begin(), input.end(), input.begin(), recurrence_functor<T>{});
}

/**
* @brief Return an array whose values y_i are the number of null entries
* in between the last valid entry of the input and the current index.
* Example: {1, NULL, 3, 4, NULL, NULL, 7}
        -> {0, 0     1, 0, 0,    1,    2}
*/
rmm::device_uvector<cudf::size_type> null_roll_up(column_view const& input,
                                                  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<cudf::size_type> output(
    input.size(), stream, rmm::mr::get_current_device_resource());

  auto device_view = *column_device_view::create(input);
  auto valid_it    = cudf::detail::make_validity_iterator(device_view);

  // TODO - not sure why two iterators produce a different result
  // Invert the null iterator
  thrust::transform(rmm::exec_policy(stream),
                    valid_it,
                    valid_it + input.size(),
                    output.begin(),
                    [] __device__(bool valid) -> bool { return 1 - valid; });

  // 0, 1, 0, 1, 1, 0 -> 0, 0, 1, 0, 0, 2
  thrust::inclusive_scan_by_key(
    rmm::exec_policy(stream), output.begin(), output.end() - 1, output.begin(), output.begin() + 1);

  return output;
}

/**
* @brief modify the source pairs that eventually yield the numerator
* and denominator to account for nan values. Pairs at nan indices
* become the identity operator (1, 0). The first pair after a nan
* value or sequence of nan values has its first element multiplied by
* N factors of beta, where N is the number of preceeding NaNs.
*
* This function requires that the null rollup be passed in as a
* precomputed device vector. This is because, as in the case with
* compute_ewma_noadjust, it is possible that the null rollup is needed
* later in the calculation for something else, and we want to avoid
* computing it twice.

*/
template <typename T>
void pair_beta_adjust(column_view const& input,
                      rmm::device_uvector<thrust::pair<T, T>>& pairs,
                      rmm::device_uvector<cudf::size_type>& nullcnt,
                      rmm::cuda_stream_view stream)
{
  using pair_type        = thrust::pair<T, T>;
  auto device_view       = *column_device_view::create(input);
  auto valid_it          = cudf::detail::make_validity_iterator(device_view);
  auto valid_and_nullcnt = thrust::make_zip_iterator(thrust::make_tuple(valid_it, nullcnt.begin()));

  thrust::transform(
    rmm::exec_policy(stream),
    valid_and_nullcnt,
    valid_and_nullcnt + input.size(),
    pairs.begin(),
    pairs.begin(),
    [] __device__(thrust::tuple<bool, int> valid_and_nullcnt, pair_type pair) -> pair_type {
      bool const valid = thrust::get<0>(valid_and_nullcnt);
      int const exp    = thrust::get<1>(valid_and_nullcnt);
      if (valid and (exp != 0)) {
        // The value is non-null, but nulls preceeded it
        // must adjust the second element of the pair
        T beta  = pair.first;
        T scale = pair.second;
        return {beta * (pow(beta, exp)), scale};
      } else if (!valid) {
        // the value is null, carry the previous value forward
        // "identity operator" is used
        return {1.0, 0.0};
      } else {
        return pair;
      }
    });
}

template <typename T>
rmm::device_uvector<T> compute_ewma_adjust(column_view const& input,
                                           T const beta,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  using pair_type = thrust::pair<T, T>;
  rmm::device_uvector<T> output(input.size(), stream, mr);
  rmm::device_uvector<pair_type> pairs(input.size(), stream);

  if (input.has_nulls()) {
    rmm::device_uvector<cudf::size_type> nullcnt = null_roll_up(input, stream);
    // Numerator
    // Fill with pairs
    thrust::transform(rmm::exec_policy(stream),
                      input.begin<T>(),
                      input.end<T>(),
                      pairs.begin(),
                      [=] __device__(T input) -> pair_type {
                        return {beta, input};
                      });

    pair_beta_adjust(input, pairs, nullcnt, stream);
    compute_recurrence(pairs, stream);

    // copy the second elements to the output for now
    thrust::transform(rmm::exec_policy(stream),
                      pairs.begin(),
                      pairs.end(),
                      output.begin(),
                      [=] __device__(pair_type pair) -> T { return pair.second; });

    // Denominator
    // Fill with pairs
    thrust::fill(rmm::exec_policy(stream), pairs.begin(), pairs.end(), pair_type(beta, 1.0));

    pair_beta_adjust(input, pairs, nullcnt, stream);
    compute_recurrence(pairs, stream);

    thrust::transform(
      rmm::exec_policy(stream),
      pairs.begin(),
      pairs.end(),
      output.begin(),
      output.begin(),
      [] __device__(pair_type pair, T numerator) -> T { return numerator / pair.second; });

  } else {
    // Numerator
    // Fill with pairs
    thrust::transform(rmm::exec_policy(stream),
                      input.begin<T>(),
                      input.end<T>(),
                      pairs.begin(),
                      [beta] __device__(T input) -> pair_type {
                        return {beta, input};
                      });
    compute_recurrence(pairs, stream);

    // copy the second elements to the output for now
    thrust::transform(rmm::exec_policy(stream),
                      pairs.begin(),
                      pairs.end(),
                      output.begin(),
                      [] __device__(pair_type pair) -> T { return pair.second; });

    // Denominator
    // Fill with pairs
    thrust::fill(rmm::exec_policy(stream), pairs.begin(), pairs.end(), pair_type(beta, 1.0));
    compute_recurrence(pairs, stream);

    thrust::transform(
      rmm::exec_policy(stream),
      pairs.begin(),
      pairs.end(),
      output.begin(),
      output.begin(),
      [] __device__(pair_type pair, T numerator) -> T { return numerator / pair.second; });
  }
  return output;
}

template <typename T>
rmm::device_uvector<T> compute_ewma_noadjust(column_view const& input,
                                             T const beta,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  using pair_type = thrust::pair<T, T>;
  rmm::device_uvector<T> output(input.size(), stream, mr);
  rmm::device_uvector<pair_type> pairs(input.size(), stream);

  thrust::transform(rmm::exec_policy(stream),
                    input.begin<T>(),
                    input.end<T>(),
                    pairs.begin(),
                    [beta] __device__(T input) -> pair_type {
                      return {beta, (1.0 - beta) * input};
                    });

  // TODO: the first pair is WRONG using the above. Reset just that pair

  thrust::transform(rmm::exec_policy(stream),
                    input.begin<T>(),
                    std::next(input.begin<T>()),
                    pairs.begin(),
                    [beta] __device__(T input) -> pair_type {
                      return {beta, input};
                    });

  if (input.has_nulls()) {
    /*
    In this case, a denominator actually has to be computed. The formula is
    y_{i+1} = (1 - alpha)x_{i-1} + alpha x_i, but really there is a "denominator"
    which is the sum of the weights: alpha + (1 - alpha) == 1. If a null is
    encountered, that means that the "previous" value is downweighted by a
    factor (for each missing value). For example this would y_2 be for one NULL:
    data = {x_0, NULL, x_1},
    y_2 = (1 - alpha)**2 x_0 + alpha * x_2 / (alpha + (1-alpha)**2)

    As such, the pairs must be updated before summing like the adjusted case to
    properly downweight the previous values. But now but we also need to compute
    the normalization factors and divide the results into them at the end.

    */

    rmm::device_uvector<cudf::size_type> nullcnt = null_roll_up(input, stream);
    pair_beta_adjust(input, pairs, nullcnt, stream);

    rmm::device_uvector<T> nullcnt_factor(nullcnt.size(), stream, mr);

    thrust::transform(rmm::exec_policy(stream),
                      nullcnt.begin(),
                      nullcnt.end(),
                      nullcnt_factor.begin(),
                      [=] __device__(T exponent) -> T {
                        // ex: 2 -> alpha + (1  - alpha)**2
                        if (exponent != 0) {
                          return (1.0 - beta) + pow(beta, exponent + 1);
                        } else {
                          return exponent;
                        }
                      });

    auto device_view = *column_device_view::create(input);
    auto valid_it    = detail::make_validity_iterator(device_view);
    auto null_and_null_count =
      thrust::make_zip_iterator(thrust::make_tuple(valid_it, nullcnt_factor.begin()));
    thrust::transform(
      rmm::exec_policy(stream),
      null_and_null_count,
      null_and_null_count + input.size(),
      pairs.begin(),
      pairs.begin(),
      [] __device__(thrust::tuple<bool, T> null_and_null_count, pair_type pair) -> pair_type {
        bool const is_valid = thrust::get<0>(null_and_null_count);
        T const factor      = thrust::get<1>(null_and_null_count);

        T const ci = pair.first;
        T const cj = pair.second;

        if (is_valid and (factor != 0.0)) {
          return {ci / factor, cj / factor};
        } else {
          return {ci, cj};
        }
      });
  }
  compute_recurrence(pairs, stream);
  // copy the second elements to the output for now
  thrust::transform(rmm::exec_policy(stream),
                    pairs.begin(),
                    pairs.end(),
                    output.begin(),
                    [] __device__(pair_type pair) -> T { return pair.second; });
  return output;
}

template <typename T>
std::unique_ptr<column> ewma(std::unique_ptr<aggregation> const& agg,
                             column_view const& input,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_floating_point(input.type()), "Column must be floating point type");
  bool adjust = (dynamic_cast<ewma_aggregation*>(agg.get()))->adjust;
  T com       = (dynamic_cast<ewma_aggregation*>(agg.get()))->com;
  // center of mass is easier for the user, but the recurrences are
  // better expressed in terms of the derived parameter `beta`
  T beta = 1.0 - (1.0 / (com + 1.0));

  rmm::device_uvector<T> data(input.size(), stream, mr);
  if (adjust) {
    data = compute_ewma_adjust(input, beta, stream, mr);
  } else {
    data = compute_ewma_noadjust(input, beta, stream, mr);
  }
  return std::make_unique<column>(
    cudf::data_type(cudf::type_to_id<T>()), input.size(), std::move(data.release()));
}

struct ewma_functor {
  template <typename T>
  std::unique_ptr<column> operator()(std::unique_ptr<aggregation> const& agg,
                                     column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return std::unique_ptr<column>();
  }
};

template <>
std::unique_ptr<column> ewma_functor::operator()<float>(std::unique_ptr<aggregation> const& agg,
                                                        column_view const& input,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::mr::device_memory_resource* mr)
{
  return ewma<float>(agg, input, stream, mr);
}

template <>
std::unique_ptr<column> ewma_functor::operator()<double>(std::unique_ptr<aggregation> const& agg,
                                                         column_view const& input,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr)
{
  return ewma<double>(agg, input, stream, mr);
}

std::unique_ptr<column> ewm(column_view const& input,
                            std::unique_ptr<aggregation> const& agg,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(input.type(), ewma_functor{}, agg, input, stream, mr);
}

}  // namespace detail
}  // namespace cudf
