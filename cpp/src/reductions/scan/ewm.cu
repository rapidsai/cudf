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
using pair_type = thrust::pair<T, T>;

/**
 * @brief functor to be summed over in a prefix sum such that
 * the recurrence in question is solved.
 * see https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf S. 1.4
 * for details
 */
template <typename T>
class recurrence_functor {
 public:
  __device__ pair_type<T> operator()(pair_type<T> ci, pair_type<T> cj)
  {
    return {ci.first * cj.first, ci.second * cj.first + cj.second};
  }
};

/**
* @brief Return an array whose values y_i are the number of null entries
* in between the last valid entry of the input and the current index.
* Example: {1, NULL, 3, 4, NULL, NULL, 7}
        -> {0, 0     1, 0, 0,    1,    2}
*/
rmm::device_uvector<cudf::size_type> null_roll_up(column_view const& input,
                                                  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<cudf::size_type> output(input.size(), stream);

  auto device_view = column_device_view::create(input);
  auto valid_it    = cudf::detail::make_validity_iterator(*device_view);

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

template <typename T>
rmm::device_uvector<T> compute_ewma_adjust(column_view const& input,
                                           T const beta,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<T> output(input.size(), stream);
  rmm::device_uvector<pair_type<T>> pairs(input.size(), stream);
  rmm::device_uvector<cudf::size_type> nullcnt(input.size(), stream);

  if (input.has_nulls()) {
    nullcnt = null_roll_up(input, stream);

    auto device_view = column_device_view::create(input);
    auto valid_it    = cudf::detail::make_validity_iterator(*device_view);
    auto valid_and_nullcnt =
      thrust::make_zip_iterator(thrust::make_tuple(valid_it, nullcnt.begin()));
    thrust::transform(
      rmm::exec_policy(stream),
      valid_and_nullcnt,
      valid_and_nullcnt + input.size(),
      input.begin<T>(),
      pairs.begin(),
      [beta] __device__(thrust::tuple<bool, int> const valid_and_nullcnt, T input) -> pair_type<T> {
        bool const valid = thrust::get<0>(valid_and_nullcnt);
        int const exp    = thrust::get<1>(valid_and_nullcnt);
        if (valid and (exp != 0)) {
          // The value is non-null, but nulls preceeded it
          // must adjust the second element of the pair

          return {beta * (pow(beta, exp)), input};
        } else if (!valid) {
          // the value is null, carry the previous value forward
          // "identity operator" is used
          return {1.0, 0.0};
        } else {
          return {beta, input};
        }
      });
  } else {
    thrust::transform(rmm::exec_policy(stream),
                      input.begin<T>(),
                      input.end<T>(),
                      pairs.begin(),
                      [=] __device__(T input) -> pair_type<T> {
                        return {beta, input};
                      });
  }
  thrust::inclusive_scan(
    rmm::exec_policy(stream), pairs.begin(), pairs.end(), pairs.begin(), recurrence_functor<T>{});

  // copy the second elements to the output for now
  thrust::transform(rmm::exec_policy(stream),
                    pairs.begin(),
                    pairs.end(),
                    output.begin(),
                    [=] __device__(pair_type<T> pair) -> T { return pair.second; });

  // Denominator
  // Fill with pairs
  thrust::fill(rmm::exec_policy(stream), pairs.begin(), pairs.end(), pair_type<T>(beta, 1.0));

  if (input.has_nulls()) {
    auto device_view = column_device_view::create(input);
    auto valid_it    = cudf::detail::make_validity_iterator(*device_view);
    auto valid_and_nullcnt =
      thrust::make_zip_iterator(thrust::make_tuple(valid_it, nullcnt.begin()));
    thrust::transform(
      rmm::exec_policy(stream),
      valid_and_nullcnt,
      valid_and_nullcnt + input.size(),
      input.begin<T>(),
      pairs.begin(),
      [beta] __device__(thrust::tuple<bool, int> const valid_and_nullcnt, T input) -> pair_type<T> {
        bool const valid = thrust::get<0>(valid_and_nullcnt);
        int const exp    = thrust::get<1>(valid_and_nullcnt);

        if (valid and (exp != 0)) {
          // The value is non-null, but nulls preceeded it
          // must adjust the second element of the pair

          return {beta * (pow(beta, exp)), 1.0};
        } else if (!valid) {
          // the value is null, carry the previous value forward
          // "identity operator" is used
          return {1.0, 0.0};
        } else {
          return {beta, 1.0};
        }
      });
  } else {
    thrust::fill(rmm::exec_policy(stream), pairs.begin(), pairs.end(), pair_type<T>(beta, 1.0));
  }

  thrust::inclusive_scan(
    rmm::exec_policy(stream), pairs.begin(), pairs.end(), pairs.begin(), recurrence_functor<T>{});

  thrust::transform(
    rmm::exec_policy(stream),
    pairs.begin(),
    pairs.end(),
    output.begin(),
    output.begin(),
    [] __device__(pair_type<T> pair, T numerator) -> T { return numerator / pair.second; });

  return output;
}

template <typename T>
rmm::device_uvector<T> compute_ewma_noadjust(column_view const& input,
                                             T const beta,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<T> output(input.size(), stream);
  rmm::device_uvector<pair_type<T>> pairs(input.size(), stream);
  rmm::device_uvector<size_type> nullcnt(input.size(), stream);

  // denominators are all 1 so dont need to be computed
  // pairs are all (beta, 1-beta x_i) except for the first one

  if (!input.has_nulls()) {
    thrust::transform(rmm::exec_policy(stream),
                      input.begin<T>(),
                      input.end<T>(),
                      thrust::make_counting_iterator<size_type>(0),
                      pairs.begin(),
                      [beta] __device__(T input, size_type index) -> pair_type<T> {
                        if (index == 0) {
                          return {beta, input};
                        } else {
                          return {beta, (1.0 - beta) * input};
                        }
                      });
  } else {
    /*
    In this case, a denominator actually has to be computed. The formula is
    y_{i+1} = (1 - alpha)x_{i-1} + alpha x_i, but really there is a "denominator"
    which is the sum of the weights: alpha + (1 - alpha) == 1. If a null is
    encountered, that means that the "previous" value is downweighted by a
    factor (for each missing value). For example this would y_2 be for one null:
    data = {x_0, NULL, x_1},
    y_2 = (1 - alpha)**2 x_0 + alpha * x_2 / (alpha + (1-alpha)**2)

    As such, the pairs must be updated before summing like the adjusted case to
    properly downweight the previous values. But now but we also need to compute
    the normalization factors and divide the results into them at the end.
    */
    nullcnt          = null_roll_up(input, stream);
    auto device_view = column_device_view::create(input);
    auto valid_it    = detail::make_validity_iterator(*device_view);

    auto data = thrust::make_zip_iterator(thrust::make_tuple(
      input.begin<T>(), thrust::make_counting_iterator<size_type>(0), valid_it, nullcnt.begin()));

    thrust::transform(
      rmm::exec_policy(stream),
      data,
      data + input.size(),
      pairs.begin(),
      [beta] __device__(thrust::tuple<T, size_type, bool, size_type> data) -> pair_type<T> {
        T input           = thrust::get<0>(data);
        size_type index   = thrust::get<1>(data);
        bool is_valid     = thrust::get<2>(data);
        size_type nullcnt = thrust::get<3>(data);

        if (index == 0) {
          return {beta, input};
        } else {
          if (is_valid and nullcnt == 0) {
            // preceeding value is valid, return normal pair
            return {beta, (1.0 - beta) * input};
          } else if (is_valid and nullcnt != 0) {
            // one or more preceeding values is null, adjust by how many
            T factor = (1.0 - beta) + pow(beta, nullcnt + 1);
            return {(beta * (pow(beta, nullcnt)) / factor), ((1.0 - beta) * input) / factor};
          } else {
            // value is not valid
            return {1.0, 0.0};
          }
        }
      });
  }

  thrust::inclusive_scan(
    rmm::exec_policy(stream), pairs.begin(), pairs.end(), pairs.begin(), recurrence_functor<T>{});

  // copy the second elements to the output for now
  thrust::transform(rmm::exec_policy(stream),
                    pairs.begin(),
                    pairs.end(),
                    output.begin(),
                    [] __device__(pair_type<T> pair) -> T { return pair.second; });
  return output;
}

template <typename T>
std::unique_ptr<column> ewma(std::unique_ptr<aggregation> const& agg,
                             column_view const& input,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_floating_point(input.type()), "Column must be floating point type");
  cudf::ewm_history history = (dynamic_cast<ewma_aggregation*>(agg.get()))->history;
  T com                     = (dynamic_cast<ewma_aggregation*>(agg.get()))->com;
  // center of mass is easier for the user, but the recurrences are
  // better expressed in terms of the derived parameter `beta`
  T beta = com / (com + 1.0);

  auto result = [&]() {
    if (history == cudf::ewm_history::INFINITE) {
      return compute_ewma_adjust(input, beta, stream, mr).release();
    } else {
      return compute_ewma_noadjust(input, beta, stream, mr).release();
    }
  }();
  return std::make_unique<column>(
    cudf::data_type(cudf::type_to_id<T>()), input.size(), std::move(result));
}

struct ewma_functor {
  template <typename T>
  std::unique_ptr<column> operator()(std::unique_ptr<aggregation> const& agg,
                                     column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Unsupported type for EWMA.");
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
