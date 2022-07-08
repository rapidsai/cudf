/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#pragma once

#include "generate_input.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <memory>
#include <type_traits>

/**
 * @brief Real Type that has atleast number of bits of integral type in its mantissa.
 *  number of bits of integrals < 23 bits of mantissa in float
 * to allow full range of integer bits to be generated.
 * @tparam T integral type
 */
template <typename T>
using integral_to_realType =
  std::conditional_t<cuda::std::is_floating_point_v<T>,
                     T,
                     std::conditional_t<sizeof(T) * 8 <= 23, float, double>>;

/**
 * @brief Generates a normal distribution between zero and upper_bound.
 */
template <typename T>
auto make_normal_dist(T lower_bound, T upper_bound)
{
  using realT    = integral_to_realType<T>;
  T const mean   = lower_bound + (upper_bound - lower_bound) / 2;
  T const stddev = (upper_bound - lower_bound) / 6;
  return thrust::random::normal_distribution<realT>(mean, stddev);
}

template <typename T, std::enable_if_t<cuda::std::is_integral_v<T>, T>* = nullptr>
auto make_uniform_dist(T range_start, T range_end)
{
  return thrust::uniform_int_distribution<T>(range_start, range_end);
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
auto make_uniform_dist(T range_start, T range_end)
{
  return thrust::uniform_real_distribution<T>(range_start, range_end);
}

template <typename T>
double geometric_dist_p(T range_size)
{
  constexpr double percentage_in_range = 0.99;
  double const p                       = 1 - exp(log(1 - percentage_in_range) / range_size);
  return p ? p : std::numeric_limits<double>::epsilon();
}

/**
 * @brief Generates a geometric distribution between lower_bound and upper_bound.
 * This distribution is an approximation generated using normal distribution.
 *
 * @tparam T Result type of the number to produce.
 */
template <typename T>
class geometric_distribution : public thrust::random::normal_distribution<integral_to_realType<T>> {
  using realType = integral_to_realType<T>;
  using super_t  = thrust::random::normal_distribution<realType>;
  T _lower_bound;
  T _upper_bound;

 public:
  using result_type = T;
  __host__ __device__ explicit geometric_distribution(T lower_bound, T upper_bound)
    : super_t(0, std::labs(upper_bound - lower_bound) / 4.0),
      _lower_bound(lower_bound),
      _upper_bound(upper_bound)
  {
  }

  template <typename UniformRandomNumberGenerator>
  __host__ __device__ result_type operator()(UniformRandomNumberGenerator& urng)
  {
    return _lower_bound < _upper_bound ? std::abs(super_t::operator()(urng)) + _lower_bound
                                       : _lower_bound - std::abs(super_t::operator()(urng));
  }
};

template <typename T, typename Generator>
struct value_generator {
  using result_type = T;

  value_generator(T lower_bound, T upper_bound, thrust::minstd_rand& engine, Generator gen)
    : lower_bound(std::min(lower_bound, upper_bound)),
      upper_bound(std::max(lower_bound, upper_bound)),
      engine(engine),
      dist(gen)
  {
  }

  __device__ T operator()(size_t n)
  {
    engine.discard(n);
    if constexpr (cuda::std::is_integral_v<T> &&
                  cuda::std::is_floating_point_v<decltype(dist(engine))>) {
      return std::clamp(static_cast<T>(std::round(dist(engine))), lower_bound, upper_bound);
    } else {
      return std::clamp(dist(engine), lower_bound, upper_bound);
    }
    // Note: uniform does not need clamp, because already range is guaranteed to be within bounds.
  }

  T lower_bound;
  T upper_bound;
  thrust::minstd_rand engine;
  Generator dist;
};

template <typename T>
using distribution_fn = std::function<rmm::device_uvector<T>(thrust::minstd_rand&, size_t)>;

template <
  typename T,
  std::enable_if_t<cuda::std::is_integral_v<T> or cuda::std::is_floating_point_v<T>, T>* = nullptr>
distribution_fn<T> make_distribution(distribution_id dist_id, T lower_bound, T upper_bound)
{
  switch (dist_id) {
    case distribution_id::NORMAL:
      return [lower_bound, upper_bound, dist = make_normal_dist(lower_bound, upper_bound)](
               thrust::minstd_rand& engine, size_t size) -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, cudf::default_stream_value);
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         value_generator{lower_bound, upper_bound, engine, dist});
        return result;
      };
    case distribution_id::UNIFORM:
      return [lower_bound, upper_bound, dist = make_uniform_dist(lower_bound, upper_bound)](
               thrust::minstd_rand& engine, size_t size) -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, cudf::default_stream_value);
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         value_generator{lower_bound, upper_bound, engine, dist});
        return result;
      };
    case distribution_id::GEOMETRIC:
      // kind of exponential distribution from lower_bound to upper_bound.
      return [lower_bound, upper_bound, dist = geometric_distribution<T>(lower_bound, upper_bound)](
               thrust::minstd_rand& engine, size_t size) -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, cudf::default_stream_value);
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         value_generator{lower_bound, upper_bound, engine, dist});
        return result;
      };
    default: CUDF_FAIL("Unsupported probability distribution");
  }
}
