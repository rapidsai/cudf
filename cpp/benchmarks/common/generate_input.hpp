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

#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <map>
#include <optional>

/**
 * @file generate_input.hpp
 * @brief Contains declarations of functions that generate columns filled with random data.
 *
 * Also includes the data profile descriptor classes.
 *
 * The create_random_table functions take a data profile, the information about table size and a
 * seed to deterministically generate a table with given parameters.
 *
 * Currently, the data generation is done on the CPU and the data is then copied to the device
 * memory.
 */

/**
 * @brief Identifies a group of related column's logical element types.
 */
enum class type_group_id : int32_t {
  INTEGRAL = static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS),
  INTEGRAL_SIGNED,
  FLOATING_POINT,
  NUMERIC,
  TIMESTAMP,
  DURATION,
  FIXED_POINT,
  COMPOUND,
  NESTED,
};

/**
 * @brief Identifies a probability distribution type.
 */
enum class distribution_id : int8_t {
  UNIFORM,    ///< Uniform sampling between the given bounds. Provides the best coverage of the
              ///< overall value range. Real data rarely has this distribution.
  NORMAL,     ///< Gaussian sampling - most samples are close to the middle of the range. Good for
              ///< simulating real-world numeric data.
  GEOMETRIC,  ///< Geometric sampling - highest chance to sample close to the lower bound. Good for
              ///< simulating real data with asymmetric distribution (unsigned values, timestamps).
};

// Default distribution types for each type
namespace {
template <typename T, std::enable_if_t<cudf::is_chrono<T>()>* = nullptr>
distribution_id default_distribution_id()
{
  return distribution_id::GEOMETRIC;
}

template <typename T, std::enable_if_t<!std::is_unsigned_v<T> && cudf::is_numeric<T>()>* = nullptr>
distribution_id default_distribution_id()
{
  return distribution_id::NORMAL;
}

template <typename T,
          std::enable_if_t<!std::is_same_v<T, bool> && std::is_unsigned_v<T> &&
                           cudf::is_numeric<T>()>* = nullptr>
distribution_id default_distribution_id()
{
  return distribution_id::GEOMETRIC;
}

/**
 * @brief Default range for the timestamp types: 1970 - 2020.
 *
 * The 2020 timestamp is used as a lower bound to bias the geometric distribution to recent
 * timestamps.
 */
template <typename T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
std::pair<int64_t, int64_t> default_range()
{
  using cuda::std::chrono::duration_cast;
  auto const year = duration_cast<typename T::duration>(cudf::duration_D{365l});
  return {50 * year.count(), 0};
}

/**
 * @brief Default range for the duration types.
 *
 * If a geometric distribution is used, it will bias towards short duration values.
 */
template <typename T, std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
std::pair<int64_t, int64_t> default_range()
{
  using cuda::std::chrono::duration_cast;
  auto const year = duration_cast<typename T::duration>(cudf::duration_D{365l});
  return {0, 2 * year.count()};
}

template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
std::pair<T, T> default_range()
{
  // Limits need to be such that `upper - lower` does not overflow
  return {std::numeric_limits<T>::lowest() / 2, std::numeric_limits<T>::max() / 2};
}
}  // namespace

/**
 * @brief Enables partial specializations with SFINAE.
 */
template <typename T, typename Enable = void>
struct distribution_params;

/**
 * @brief Numeric values are parameterized with a distribution type and bounds of the same type.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<!std::is_same_v<T, bool> && cudf::is_numeric<T>()>> {
  distribution_id id;
  T lower_bound;
  T upper_bound;
};

/**
 * @brief Booleans are parameterized with the probability of getting `true` value.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, bool>>> {
  double probability_true;
};

/**
 * @brief Timestamps and durations are parameterized with a distribution type and int64_t bounds.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<cudf::is_chrono<T>()>> {
  distribution_id id;
  int64_t lower_bound;
  int64_t upper_bound;
};

/**
 * @brief Strings are parameterized by the distribution of their length, as an integral value.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>> {
  distribution_params<uint32_t> length_params;
};

/**
 * @brief Lists are parameterized by the distribution of their length, maximal nesting level, and
 * the element type.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, cudf::list_view>>> {
  cudf::type_id element_type;
  distribution_params<uint32_t> length_params;
  cudf::size_type max_depth;
};

/**
 * @brief Structs are parameterized by the maximal nesting level, and the leaf column types.
 */
template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, cudf::struct_view>>> {
  std::vector<cudf::type_id> leaf_types;
  cudf::size_type max_depth;
};

// Present for compilation only. To be implemented once reader/writers support the fixed width type.
template <typename T>
struct distribution_params<T, std::enable_if_t<cudf::is_fixed_point<T>()>> {
};

/**
 * @brief Returns a vector of types, corresponding to the input type or a type group.
 *
 * If the input is a `cudf::type_id` enumerator, function simply returns a vector containing this
 * type. If the input value corresponds to a `type_group_id` enumerator, function returns a vector
 * containing all types in the input group.
 *
 * @param id Integer equal to either a `cudf::type_id` enumerator or a `type_group_id` enumerator.
 */
std::vector<cudf::type_id> get_type_or_group(int32_t id);

/**
 * @brief Returns a vector of types, corresponding to the input types or type groups.
 *
 * If an element of the input vector is a `cudf::type_id` enumerator, function return value simply
 * includes this type. If an element of the input vector is a `type_group_id` enumerator, function
 * return value includes all types corresponding to the group enumerator.
 *
 * @param ids Vector of integers equal to either a `cudf::type_id` enumerator or a `type_group_id`
 * enumerator.
 */
std::vector<cudf::type_id> get_type_or_group(std::vector<int32_t> const& ids);

/**
 * @brief Contains data parameters for all types.
 *
 * This class exposes APIs to set and get distribution parameters for each supported type.
 * Parameters can be set for multiple types with a single call by passing a `type_group_id` instead
 * of `cudf::type_id`.
 *
 * All types have default parameters so it's not necessary to set the parameters before using them.
 */
class data_profile {
  std::map<cudf::type_id, distribution_params<uint64_t>> int_params;
  std::map<cudf::type_id, distribution_params<double>> float_params;
  distribution_params<cudf::string_view> string_dist_desc{{distribution_id::NORMAL, 0, 32}};
  distribution_params<cudf::list_view> list_dist_desc{
    cudf::type_id::INT32, {distribution_id::GEOMETRIC, 0, 100}, 2};
  distribution_params<cudf::struct_view> struct_dist_desc{
    {cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::STRING}, 2};
  std::map<cudf::type_id, distribution_params<__uint128_t>> decimal_params;

  double bool_probability              = 0.5;
  std::optional<double> null_frequency = 0.01;
  cudf::size_type cardinality          = 2000;
  cudf::size_type avg_run_length       = 4;

 public:
  template <typename T,
            std::enable_if_t<!std::is_same_v<T, bool> && cuda::std::is_integral_v<T>, T>* = nullptr>
  distribution_params<T> get_distribution_params() const
  {
    auto it = int_params.find(cudf::type_to_id<T>());
    if (it == int_params.end()) {
      auto const range = default_range<T>();
      return distribution_params<T>{default_distribution_id<T>(), range.first, range.second};
    } else {
      auto& desc = it->second;
      return {desc.id, static_cast<T>(desc.lower_bound), static_cast<T>(desc.upper_bound)};
    }
  }

  template <typename T, std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
  distribution_params<T> get_distribution_params() const
  {
    auto it = float_params.find(cudf::type_to_id<T>());
    if (it == float_params.end()) {
      auto const range = default_range<T>();
      return distribution_params<T>{default_distribution_id<T>(), range.first, range.second};
    } else {
      auto& desc = it->second;
      return {desc.id, static_cast<T>(desc.lower_bound), static_cast<T>(desc.upper_bound)};
    }
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, bool>>* = nullptr>
  distribution_params<T> get_distribution_params() const
  {
    return distribution_params<T>{bool_probability};
  }

  template <typename T, std::enable_if_t<cudf::is_chrono<T>()>* = nullptr>
  distribution_params<T> get_distribution_params() const
  {
    auto it = int_params.find(cudf::type_to_id<T>());
    if (it == int_params.end()) {
      auto const range = default_range<T>();
      return distribution_params<T>{default_distribution_id<T>(), range.first, range.second};
    } else {
      auto& desc = it->second;
      return {
        desc.id, static_cast<int64_t>(desc.lower_bound), static_cast<int64_t>(desc.upper_bound)};
    }
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>* = nullptr>
  distribution_params<T> get_distribution_params() const
  {
    return string_dist_desc;
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::list_view>>* = nullptr>
  distribution_params<T> get_distribution_params() const
  {
    return list_dist_desc;
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::struct_view>>* = nullptr>
  distribution_params<T> get_distribution_params() const
  {
    return struct_dist_desc;
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  distribution_params<typename T::rep> get_distribution_params() const
  {
    using rep = typename T::rep;
    auto it   = decimal_params.find(cudf::type_to_id<T>());
    if (it == decimal_params.end()) {
      auto const range = default_range<rep>();
      return distribution_params<rep>{default_distribution_id<rep>(), range.first, range.second};
    } else {
      auto& desc = it->second;
      return {desc.id, static_cast<rep>(desc.lower_bound), static_cast<rep>(desc.upper_bound)};
    }
  }

  auto get_bool_probability() const { return bool_probability; }
  auto get_null_frequency() const { return null_frequency; };
  [[nodiscard]] auto get_cardinality() const { return cardinality; };
  [[nodiscard]] auto get_avg_run_length() const { return avg_run_length; };

  // Users should pass integral values for bounds when setting the parameters for types that have
  // discrete distributions (integers, strings, lists). Otherwise the call with have no effect.
  template <typename T,
            typename Type_enum,
            std::enable_if_t<cuda::std::is_integral_v<T>, T>* = nullptr>
  void set_distribution_params(Type_enum type_or_group,
                               distribution_id dist,
                               T lower_bound,
                               T upper_bound)
  {
    for (auto tid : get_type_or_group(static_cast<int32_t>(type_or_group))) {
      if (tid == cudf::type_id::STRING) {
        string_dist_desc.length_params = {
          dist, static_cast<uint32_t>(lower_bound), static_cast<uint32_t>(upper_bound)};
      } else if (tid == cudf::type_id::LIST) {
        list_dist_desc.length_params = {
          dist, static_cast<uint32_t>(lower_bound), static_cast<uint32_t>(upper_bound)};
      } else {
        int_params[tid] = {
          dist, static_cast<uint64_t>(lower_bound), static_cast<uint64_t>(upper_bound)};
      }
    }
  }

  // Users should pass floating point values for bounds when setting the parameters for types that
  // have continuous distributions (floating point types). Otherwise the call with have no effect.
  template <typename T,
            typename Type_enum,
            std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
  void set_distribution_params(Type_enum type_or_group,
                               distribution_id dist,
                               T lower_bound,
                               T upper_bound)
  {
    for (auto tid : get_type_or_group(static_cast<int32_t>(type_or_group))) {
      float_params[tid] = {
        dist, static_cast<double>(lower_bound), static_cast<double>(upper_bound)};
    }
  }

  template <typename T, typename Type_enum, std::enable_if_t<cudf::is_chrono<T>(), T>* = nullptr>
  void set_distribution_params(Type_enum type_or_group,
                               distribution_id dist,
                               typename T::rep lower_bound,
                               typename T::rep upper_bound)
  {
    for (auto tid : get_type_or_group(static_cast<int32_t>(type_or_group))) {
      int_params[tid] = {
        dist, static_cast<uint64_t>(lower_bound), static_cast<uint64_t>(upper_bound)};
    }
  }

  void set_bool_probability(double p) { bool_probability = p; }
  void set_null_frequency(std::optional<double> f) { null_frequency = f; }
  void set_cardinality(cudf::size_type c) { cardinality = c; }
  void set_avg_run_length(cudf::size_type avg_rl) { avg_run_length = avg_rl; }

  void set_list_depth(cudf::size_type max_depth)
  {
    CUDF_EXPECTS(max_depth > 0, "List depth must be positive");
    list_dist_desc.max_depth = max_depth;
  }

  void set_list_type(cudf::type_id type) { list_dist_desc.element_type = type; }

  void set_struct_depth(cudf::size_type max_depth)
  {
    CUDF_EXPECTS(max_depth > 0, "Struct depth must be positive");
    struct_dist_desc.max_depth = max_depth;
  }

  void set_struct_types(std::vector<cudf::type_id> const& types)
  {
    CUDF_EXPECTS(
      std::none_of(
        types.cbegin(), types.cend(), [](auto& type) { return type == cudf::type_id::STRUCT; }),
      "Cannot include STRUCT as its own subtype");
    struct_dist_desc.leaf_types = types;
  }
};

/**
 * @brief Strongly typed table size in bytes. Used to disambiguate overloads of
 * `create_random_table`.
 */
struct table_size_bytes {
  size_t size;
};

/**
 * @brief Strongly typed row count. Used to disambiguate overloads of `create_random_table`.
 */
struct row_count {
  cudf::size_type count;
};

/**
 * @brief Deterministically generates a table filled with data with the given parameters.
 *
 * @param dtype_ids Vector of requested column types
 * @param table_bytes Target size of the output table, in bytes. Some type may not produce columns
 * of exact size
 * @param data_params optional, set of data parameters describing the data profile for each type
 * @param seed optional, seed for the pseudo-random engine
 */
std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 table_size_bytes table_bytes,
                                                 data_profile const& data_params = data_profile{},
                                                 unsigned seed                   = 1);

/**
 * @brief Deterministically generates a table filled with data with the given parameters.
 *
 * @param dtype_ids Vector of requested column types
 * @param num_rows Number of rows in the output table
 * @param data_params optional, set of data parameters describing the data profile for each type
 * @param seed optional, seed for the pseudo-random engine
 */
std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 row_count num_rows,
                                                 data_profile const& data_params = data_profile{},
                                                 unsigned seed                   = 1);

/**
 * @brief Generate sequence columns starting with value 0 in first row and increasing by 1 in
 * subsequent rows.
 *
 * @param dtype_ids Vector of requested column types
 * @param num_rows Number of rows in the output table
 * @param null_probability optional, probability of a null value
 *  no value implies no null mask, =0 implies all valids, >=1 implies all nulls
 * @param seed optional, seed for the pseudo-random engine
 * @return A table with the sequence columns.
 */
std::unique_ptr<cudf::table> create_sequence_table(
  std::vector<cudf::type_id> const& dtype_ids,
  row_count num_rows,
  std::optional<double> null_probability = std::nullopt,
  unsigned seed                          = 1);

/**
 * @brief Repeats the input data types cyclically to fill a vector of @ref num_cols
 * elements.
 *
 * @param dtype_ids Vector of requested column types
 * @param num_cols Number of types in the output vector
 * @return A vector of type_ids
 */
std::vector<cudf::type_id> cycle_dtypes(std::vector<cudf::type_id> const& dtype_ids,
                                        cudf::size_type num_cols);
/**
 * @brief Create a random null mask object
 *
 * @param size number of rows
 * @param null_probability probability of a null value
 *  no value implies no null mask, =0 implies all valids, >=1 implies all nulls
 * @param seed optional, seed for the pseudo-random engine
 * @return null mask device buffer with random null mask data and null count
 */
std::pair<rmm::device_buffer, cudf::size_type> create_random_null_mask(
  cudf::size_type size, std::optional<double> null_probability = std::nullopt, unsigned seed = 1);
