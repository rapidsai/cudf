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

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
auto make_normal_dist(T range_start, T range_end)
{
  using uT            = typename std::make_unsigned<T>::type;
  uT const range_size = range_end - range_start;
  return std::binomial_distribution<uT>(range_size, 0.5);
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
auto make_normal_dist(T range_start, T range_end)
{
  T const mean   = range_start / 2 + range_end / 2;
  T const stddev = range_end / 6 - range_start / 6;
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

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
auto make_geometric_dist(T range_start, T range_end)
{
  using uT                             = typename std::make_unsigned<T>::type;
  uT const range_size                  = range_end - range_start;
  constexpr double percentage_in_range = 0.99;
  double const p                       = 1 - exp(log(1 - percentage_in_range) / range_size);
  return std::geometric_distribution<T>(p);
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
auto make_geometric_dist(T range_start, T range_end)
{
  long double const range_size         = range_end - range_start;
  constexpr double percentage_in_range = 0.99;
  double const p                       = 1 - exp(log(1 - percentage_in_range) / range_size);
  return std::exponential_distribution<T>(p);
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
std::function<T(std::mt19937&)> make_rand_generator(rand_dist_id dist_type,
                                                    T lower_bound,
                                                    T upper_bound)
{
  switch (dist_type) {
    case rand_dist_id::NORMAL:
      return [lower_bound, dist = make_normal_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T { return dist(engine) - lower_bound; };
    case rand_dist_id::UNIFORM:
      return [dist = make_uniform_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T { return dist(engine); };
    case rand_dist_id::GEOMETRIC:
      return [lower_bound, upper_bound, dist = make_geometric_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T {
        if (lower_bound <= upper_bound)
          return dist(engine);
        else
          return lower_bound - dist(engine) + lower_bound;
      };
    default: CUDF_FAIL("Unsupported random distribution");
  }
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
std::function<T(std::mt19937&)> make_rand_generator(rand_dist_id dist_type,
                                                    T lower_bound,
                                                    T upper_bound)
{
  switch (dist_type) {
    case rand_dist_id::NORMAL:
      return [dist = make_normal_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T { return dist(engine); };
    case rand_dist_id::UNIFORM:
      return [dist = make_uniform_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T { return dist(engine); };
    case rand_dist_id::GEOMETRIC:
      return [lower_bound, upper_bound, dist = make_geometric_dist(lower_bound, upper_bound)](
               std::mt19937& engine) mutable -> T {
        if (lower_bound <= upper_bound)
          return dist(engine);
        else
          return (lower_bound - dist(engine)) + lower_bound;
      };
    default: CUDF_FAIL("Unsupported random distribution");
  }
}
