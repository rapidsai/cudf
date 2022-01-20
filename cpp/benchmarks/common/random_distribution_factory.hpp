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

#include "generate_benchmark_input.hpp"

#include <memory>
#include <random>

/**
 * @brief Generates a normal(binomial) distribution between zero and upper_bound.
 */
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
auto make_normal_dist(T upper_bound)
{
  using uT = typename std::make_unsigned<T>::type;
  return std::binomial_distribution<uT>(upper_bound, 0.5);
}

/**
 * @brief Generates a normal distribution between zero and upper_bound.
 */
template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
auto make_normal_dist(T upper_bound)
{
  T const mean   = upper_bound / 2;
  T const stddev = upper_bound / 6;
  return std::normal_distribution<T>(mean, stddev);
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
auto make_uniform_dist(T range_start, T range_end)
{
  return std::uniform_int_distribution<T>(range_start, range_end);
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
auto make_uniform_dist(T range_start, T range_end)
{
  return std::uniform_real_distribution<T>(range_start, range_end);
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

template <typename T>
using distribution_fn = std::function<T(std::mt19937&)>;

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
distribution_fn<T> make_distribution(distribution_id did, T lower_bound, T upper_bound)
{
  switch (did) {
    case distribution_id::NORMAL:
      return [lower_bound, dist = make_normal_dist(upper_bound - lower_bound)](
               std::mt19937& engine) mutable -> T { return dist(engine) + lower_bound; };
    case distribution_id::UNIFORM:
      return [dist = make_uniform_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T { return dist(engine); };
    case distribution_id::GEOMETRIC:
      return [lower_bound, upper_bound, dist = make_geometric_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T {
        if (lower_bound <= upper_bound)
          return dist(engine);
        else
          return lower_bound - dist(engine) + lower_bound;
      };
    default: CUDF_FAIL("Unsupported probability distribution");
  }
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
distribution_fn<T> make_distribution(distribution_id dist_id, T lower_bound, T upper_bound)
{
  switch (dist_id) {
    case distribution_id::NORMAL:
      return [lower_bound, dist = make_normal_dist(upper_bound - lower_bound)](
               std::mt19937& engine) mutable -> T { return dist(engine) + lower_bound; };
    case distribution_id::UNIFORM:
      return [dist = make_uniform_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T { return dist(engine); };
    case distribution_id::GEOMETRIC:
      return [lower_bound, upper_bound, dist = make_geometric_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T {
        if (lower_bound <= upper_bound)
          return lower_bound + dist(engine);
        else
          return lower_bound - dist(engine);
      };
    default: CUDF_FAIL("Unsupported random distribution");
  }
}
