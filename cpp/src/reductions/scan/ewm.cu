/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <thrust/transform_scan.h>

namespace cudf {
namespace detail {

template <typename T>
using pair_type = thrust::pair<T, T>;

/**
 * @brief functor to be summed over in a prefix sum such that
 * the recurrence in question is solved. See
 * G. E. Blelloch. Prefix sums and their applications. Technical Report
 * CMU-CS-90-190, Nov. 1990. S. 1.4
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

template <typename T>
class ewma_functor_base {
 public:
  T beta;
  ewma_functor_base(T beta) : beta{beta} {}

  pair_type<T> IDENTITY = {1.0, 0.0};
};

template <typename T, bool nulls, bool is_numerator>
class ewma_adjust_functor : public ewma_functor_base<T> {
  using ewma_functor_base<T>::ewma_functor_base;

 public:
  using tupletype = std::conditional_t<nulls, thrust::tuple<bool, int, T>, T>;

  __device__ pair_type<T> operator()(tupletype const data)
  {
    if constexpr (nulls) {
      bool const valid = thrust::get<0>(data);
      int const exp    = thrust::get<1>(data);
      T const input    = thrust::get<2>(data);
      T const beta     = this->beta;

      if (!valid) { return this->IDENTITY; }

      T const second = [=]() {
        if constexpr (is_numerator) {
          return input;
        } else {
          return 1;
        }
      }();
      if (valid and (exp != 0)) {
        // The value is non-null, but nulls preceding it
        // must adjust the second element of the pair

        return {beta * (pow(beta, exp)), second};
      } else {
        return {beta, second};
      }
    } else {
      if constexpr (is_numerator) {
        return {this->beta, data};
      } else {
        return {this->beta, 1.0};
      }
    }
  }
};

template <typename T, bool nulls>
class ewma_noadjust_functor : public ewma_functor_base<T> {
  using ewma_functor_base<T>::ewma_functor_base;

 public:
  using tupletype = std::
    conditional_t<nulls, thrust::tuple<T, size_type, bool, size_type>, thrust::tuple<T, size_type>>;

  __device__ pair_type<T> operator()(tupletype const data)
  {
    T const beta          = this->beta;
    size_type const index = thrust::get<1>(data);
    T const input         = thrust::get<0>(data);

    if constexpr (!nulls) {
      if (index == 0) {
        return {beta, input};
      } else {
        return {beta, (1.0 - beta) * input};
      }
    } else {
      bool const is_valid     = thrust::get<2>(data);
      size_type const nullcnt = thrust::get<3>(data);

      if (index == 0) {
        return {beta, input};
      } else {
        if (is_valid and nullcnt == 0) {
          // preceding value is valid, return normal pair
          return {beta, (1.0 - beta) * input};
        } else if (is_valid and nullcnt != 0) {
          // one or more preceding values is null, adjust by how many
          T const factor = (1.0 - beta) + pow(beta, nullcnt + 1);
          return {(beta * (pow(beta, nullcnt)) / factor), ((1.0 - beta) * input) / factor};
        } else {
          // value is not valid
          return this->IDENTITY;
        }
      }
    }
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

  // null mask {0, 1, 0, 1, 1, 0} leads to output array {0, 0, 1, 0, 0, 2}
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                output.begin(),
                                std::prev(output.end()),
                                output.begin(),
                                std::next(output.begin()));
  return output;
}

template <typename T>
rmm::device_uvector<T> compute_ewma_adjust(column_view const& input,
                                           T const beta,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> output(input.size(), stream);
  rmm::device_uvector<pair_type<T>> pairs(input.size(), stream);
  rmm::device_uvector<cudf::size_type> nullcnt =
    [&input, stream]() -> rmm::device_uvector<cudf::size_type> {
    if (input.has_nulls()) {
      return null_roll_up(input, stream);
    } else {
      return rmm::device_uvector<cudf::size_type>(input.size(), stream);
    }
  }();
  if (input.has_nulls()) {
    auto device_view = column_device_view::create(input);
    auto valid_it    = cudf::detail::make_validity_iterator(*device_view);
    auto data =
      thrust::make_zip_iterator(thrust::make_tuple(valid_it, nullcnt.begin(), input.begin<T>()));

    thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                     data,
                                     data + input.size(),
                                     pairs.begin(),
                                     ewma_adjust_functor<T, true, true>{beta},
                                     recurrence_functor<T>{});

  } else {
    thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                     input.begin<T>(),
                                     input.end<T>(),
                                     pairs.begin(),
                                     ewma_adjust_functor<T, false, true>{beta},
                                     recurrence_functor<T>{});
  }

  // copy the second elements to the output for now
  thrust::transform(rmm::exec_policy(stream),
                    pairs.begin(),
                    pairs.end(),
                    output.begin(),
                    [=] __device__(pair_type<T> pair) -> T { return pair.second; });

  // denominator
  if (input.has_nulls()) {
    auto device_view = column_device_view::create(input);
    auto valid_it    = cudf::detail::make_validity_iterator(*device_view);
    auto data =
      thrust::make_zip_iterator(thrust::make_tuple(valid_it, nullcnt.begin(), input.begin<T>()));

    thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                     data,
                                     data + input.size(),
                                     pairs.begin(),
                                     ewma_adjust_functor<T, true, false>{beta},
                                     recurrence_functor<T>{});
  } else {
    auto itr = thrust::make_counting_iterator<size_type>(0);
    thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                     itr,
                                     itr + input.size(),
                                     pairs.begin(),
                                     ewma_adjust_functor<T, false, false>{beta},
                                     recurrence_functor<T>{});
  }

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
                                             rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> output(input.size(), stream);
  rmm::device_uvector<pair_type<T>> pairs(input.size(), stream);
  rmm::device_uvector<cudf::size_type> nullcnt =
    [&input, stream]() -> rmm::device_uvector<cudf::size_type> {
    if (input.has_nulls()) {
      return null_roll_up(input, stream);
    } else {
      return rmm::device_uvector<cudf::size_type>(input.size(), stream);
    }
  }();
  // denominators are all 1 and do not need to be computed
  // pairs are all (beta, 1-beta x_i) except for the first one

  if (!input.has_nulls()) {
    auto data = thrust::make_zip_iterator(
      thrust::make_tuple(input.begin<T>(), thrust::make_counting_iterator<size_type>(0)));
    thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                     data,
                                     data + input.size(),
                                     pairs.begin(),
                                     ewma_noadjust_functor<T, false>{beta},
                                     recurrence_functor<T>{});

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
    auto device_view = column_device_view::create(input);
    auto valid_it    = detail::make_validity_iterator(*device_view);

    auto data = thrust::make_zip_iterator(thrust::make_tuple(
      input.begin<T>(), thrust::make_counting_iterator<size_type>(0), valid_it, nullcnt.begin()));

    thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                     data,
                                     data + input.size(),
                                     pairs.begin(),
                                     ewma_noadjust_functor<T, true>{beta},
                                     recurrence_functor<T>());
  }

  // copy the second elements to the output for now
  thrust::transform(rmm::exec_policy(stream),
                    pairs.begin(),
                    pairs.end(),
                    output.begin(),
                    [] __device__(pair_type<T> pair) -> T { return pair.second; });
  return output;
}

template <typename T>
std::unique_ptr<column> ewma(scan_aggregation const& agg,
                             column_view const& input,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto ewma_agg = (dynamic_cast<ewma_aggregation const*>(&agg));
  if (ewma_agg == NULL) { CUDF_FAIL("Expected an EWMA aggregation."); }
  CUDF_EXPECTS(cudf::is_floating_point(input.type()), "Column must be floating point type");

  cudf::ewm_history const history = ewma_agg->history;
  T const center_of_mass          = ewma_agg->center_of_mass;

  // center of mass is easier for the user, but the recurrences are
  // better expressed in terms of the derived parameter `beta`
  T const beta = center_of_mass / (center_of_mass + 1.0);

  auto result = [&]() {
    if (history == cudf::ewm_history::INFINITE) {
      return compute_ewma_adjust(input, beta, stream, mr).release();
    } else {
      return compute_ewma_noadjust(input, beta, stream, mr).release();
    }
  }();
  return std::make_unique<column>(cudf::data_type(cudf::type_to_id<T>()),
                                  input.size(),
                                  std::move(result),
                                  rmm::device_buffer{},
                                  0);
}

struct ewma_functor {
  template <typename T>
  std::enable_if_t<!is_floating_point<T>(), std::unique_ptr<column>> operator()(
    scan_aggregation const& agg,
    column_view const& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    CUDF_FAIL("Unsupported type for EWMA.");
  }

  template <typename T>
  std::enable_if_t<is_floating_point<T>(), std::unique_ptr<column>> operator()(
    scan_aggregation const& agg,
    column_view const& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    return ewma<T>(agg, input, stream, mr);
  }
};

std::unique_ptr<column> exponentially_weighted_moving_average(column_view const& input,
                                                              scan_aggregation const& agg,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  return type_dispatcher(input.type(), ewma_functor{}, agg, input, stream, mr);
}

}  // namespace detail
}  // namespace cudf
