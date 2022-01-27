/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <rmm/device_uvector.hpp>
#include <type_traits>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/tabulate.h>
/**
 * @brief Generates a normal(binomial) distribution between zero and upper_bound.
 */
template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value or std::is_same_v<__int128_t, T>,
                                    T>* = nullptr>
auto make_normal_dist(T upper_bound)
{
  // Provided n is large enough, Normal(μ,σ2) is a good approximation for Binomial(n, p)
  // where μ = np and σ2 = np (1 - p).
  using realT        = std::conditional_t<sizeof(T) * 8 <= 23, float, double>;
  realT const mean   = static_cast<realT>(upper_bound) / 2;             // μ = np, p=0.5
  realT const stddev = std::sqrt(static_cast<realT>(upper_bound) / 4);  // sqrt(np (1 - p))
  // std::cout << "mean: " << mean << " stddev: " << stddev << " upper_bound: " << upper_bound <<
  // std::endl;
  return thrust::random::normal_distribution<realT>(mean, stddev);
}

/**
 * @brief Generates a normal distribution between zero and upper_bound.
 */
template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
auto make_normal_dist(T upper_bound)
{
  T const mean   = upper_bound / 2;
  T const stddev = upper_bound / 6;
  return thrust::random::normal_distribution<T>(mean, stddev);
}

template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value or std::is_same_v<__int128_t, T>,
                                    T>* = nullptr>
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

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
auto make_geometric_dist(T range_start, T range_end)
{
  using uT = typename std::make_unsigned<T>::type;
  if (range_start > range_end) std::swap(range_start, range_end);

  uT const range_size = (uT)range_end - (uT)range_start;
  return std::geometric_distribution<T>(geometric_dist_p(range_size));
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
auto make_geometric_dist(T range_start, T range_end)
{
  long double const range_size = range_end - range_start;
  return std::exponential_distribution<T>(geometric_dist_p(range_size));
}

template <typename T, typename Generator>
struct value_generator {
  using result_type = T;

  value_generator(T lower_bound, T upper_bound, thrust::minstd_rand& engine, Generator gen)
    : lower_bound(lower_bound), upper_bound(upper_bound), engine(engine), dist(gen)
  {
  }

  __device__ T operator()(size_t n)
  {
    engine.discard(n);
    return dist(engine) + lower_bound;
  }

  T lower_bound;
  T upper_bound;
  thrust::minstd_rand engine;
  Generator dist;
};

template <typename T, typename Generator>
struct abs_value_generator : value_generator<T, Generator> {
  abs_value_generator(T lower_bound, T upper_bound, thrust::minstd_rand& engine, Generator gen)
    : value_generator<T, Generator>::value_generator{lower_bound, upper_bound, engine, gen}
  {
  }

  __device__ T operator()(size_t n)
  {
    value_generator<T, Generator>::engine.discard(n);
    return abs(value_generator<T, Generator>::dist(value_generator<T, Generator>::engine)) +
           value_generator<T, Generator>::lower_bound;
  }
};

template <typename T>
using distribution_fn = std::function<rmm::device_uvector<T>(thrust::minstd_rand&, size_t)>;

template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value or std::is_same_v<__int128_t, T>,
                                    T>* = nullptr>
distribution_fn<T> make_distribution(distribution_id did, T lower_bound, T upper_bound)
{
  switch (did) {
    case distribution_id::NORMAL:
      return [lower_bound, upper_bound, dist = make_normal_dist(upper_bound - lower_bound)](
               thrust::minstd_rand& engine, size_t size) mutable -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, rmm::cuda_stream_default);
        std::cout << "Normal here" << std::endl;
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         value_generator{lower_bound, upper_bound, engine, dist});
        std::cout << "Tabulate done" << std::endl;
        return result;
      };
    case distribution_id::UNIFORM:
      return [dist = make_uniform_dist(lower_bound, upper_bound)](
               thrust::minstd_rand& engine, size_t size) mutable -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, rmm::cuda_stream_default);
        std::cout << "Uniform here" << std::endl;
        thrust::tabulate(
          thrust::device, result.begin(), result.end(), value_generator{T{0}, T{0}, engine, dist});
        std::cout << "Tabulate done" << std::endl;
        return result;
      };
    case distribution_id::GEOMETRIC:
      return [lower_bound, upper_bound, dist = make_normal_dist(upper_bound - lower_bound)](
               thrust::minstd_rand& engine, size_t size) mutable -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, rmm::cuda_stream_default);
        std::cout << "Geometric here" << std::endl;
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         abs_value_generator{lower_bound, upper_bound, engine, dist});
        std::cout << "Tabulate done" << std::endl;
        return result;
      };
    default: CUDF_FAIL("Unsupported probability distribution");
  }
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
distribution_fn<T> make_distribution(distribution_id dist_id, T lower_bound, T upper_bound)
{
  switch (dist_id) {
    case distribution_id::NORMAL:
      return [lower_bound, upper_bound, dist = make_normal_dist(upper_bound - lower_bound)](
               thrust::minstd_rand& engine, size_t size) mutable -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, rmm::cuda_stream_default);
        std::cout << "Normal float" << std::endl;
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         value_generator{upper_bound, lower_bound, engine, dist});
        std::cout << "Tabulate float" << std::endl;
        return result;
      };
    case distribution_id::UNIFORM:
      return [dist = make_uniform_dist(lower_bound, upper_bound)](
               thrust::minstd_rand& engine, size_t size) mutable -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, rmm::cuda_stream_default);
        std::cout << "Uniform float" << std::endl;
        thrust::tabulate(
          thrust::device, result.begin(), result.end(), value_generator{T{0}, T{0}, engine, dist});
        std::cout << "Tabulate float" << std::endl;
        return result;
      };
    case distribution_id::GEOMETRIC:
      // kind of exponential distribution from lower_bound to upper_bound.
      return [lower_bound, upper_bound, dist = make_normal_dist(upper_bound - lower_bound)](
               thrust::minstd_rand& engine, size_t size) mutable -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, rmm::cuda_stream_default);
        std::cout << "Geometric float" << std::endl;
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         abs_value_generator{lower_bound, upper_bound, engine, dist});
        std::cout << "Tabulate float" << std::endl;
        return result;
      };
    default: CUDF_FAIL("Unsupported probability distribution");
  }
}
