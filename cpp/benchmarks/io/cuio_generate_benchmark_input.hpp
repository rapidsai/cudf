#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <memory>
#include <random>
#include <vector>

template <typename T>
constexpr auto stddev()
{
  // Columns of wider types -> larger range of values
  return 1l << (sizeof(T) * 4);
}

// Type trait for  cudf timestamp types
template <typename>
struct is_timestamp {
  constexpr static bool value = false;
};

template <typename T>
struct is_timestamp<cudf::detail::timestamp<T>> {
  constexpr static bool value = true;
};

// nanoseconds in the type's unit
template <typename Ts>
constexpr int64_t nanoseconds();

template <>
constexpr int64_t nanoseconds<cudf::timestamp_ns>()
{
  return 1;
}

template <>
constexpr int64_t nanoseconds<cudf::timestamp_us>()
{
  return 1000l;
}

template <>
constexpr int64_t nanoseconds<cudf::timestamp_ms>()
{
  return 1000l * 1000;
}

template <>
constexpr int64_t nanoseconds<cudf::timestamp_s>()
{
  return 1000l * 1000 * 1000;
}

template <>
constexpr int64_t nanoseconds<cudf::timestamp_D>()
{
  return 24l * 60 * 60 * nanoseconds<cudf::timestamp_s>();
}

auto& deterministic_engine()
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  return engine;
}

// implementation for timestamp types
template <typename T, std::enable_if_t<is_timestamp<T>::value, int> = 0>
T random_element()
{
  // Timestamp for June 2020
  static constexpr int64_t current_ns    = 1591053936l * nanoseconds<cudf::timestamp_s>();
  static constexpr auto timestamp_spread = 1. / (2 * 365 * 24 * 60 * 60);  // one in two years

  static std::geometric_distribution<int64_t> seconds_gen{timestamp_spread};
  static std::uniform_int_distribution<int64_t> nanoseconds_gen{0,
                                                                nanoseconds<cudf::timestamp_s>()};

  // Most timestamps will corespond to recent years
  auto const timestamp_ns = current_ns -
                            seconds_gen(deterministic_engine()) * nanoseconds<cudf::timestamp_s>() -
                            nanoseconds_gen(deterministic_engine());

  return T(timestamp_ns / nanoseconds<T>());  // convert to the type precision
}

// implementation for numeric types
template <typename T, std::enable_if_t<not is_timestamp<T>::value, int> = 0>
T random_element()
{
  static constexpr T lower_bound = std::numeric_limits<T>::lowest();
  static constexpr T upper_bound = std::numeric_limits<T>::max();

  static std::normal_distribution<> gaussian{0., stddev<T>()};

  auto elem = gaussian(deterministic_engine());
  if (lower_bound >= 0) elem = abs(elem);
  elem = std::max(std::min(elem, (double)upper_bound), (double)lower_bound);

  return T(elem);
}

template <>
bool random_element<bool>()
{
  static std::uniform_int_distribution<> uniform{0, 1};
  return uniform(deterministic_engine()) == 1;
}

template <typename T>
std::unique_ptr<cudf::column> create_random_column(cudf::size_type col_bytes, bool include_validity)
{
  const cudf::size_type num_rows = col_bytes / sizeof(T);

  // Every 100th element is invalid
  // TODO: should this also be random?
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 100 == 0 ? false : true; });

  cudf::test::fixed_width_column_wrapper<T> wrapped_col;
  auto rand_elements =
    cudf::test::make_counting_transform_iterator(0, [](auto row) { return random_element<T>(); });
  if (include_validity) {
    wrapped_col =
      cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows, valids);
  } else {
    wrapped_col =
      cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows);
  }
  auto col = wrapped_col.release();
  col->has_nulls();
  return col;
}

template <>
std::unique_ptr<cudf::column> create_random_column<std::string>(cudf::size_type col_bytes,
                                                                bool include_validity)
{
  static constexpr int avg_string_len = 16;
  const cudf::size_type num_rows      = col_bytes / avg_string_len;

  std::poisson_distribution<> dist(avg_string_len);

  std::vector<int32_t> offsets{0};
  offsets.reserve(num_rows + 1);
  for (int i = 1; i < num_rows; ++i) {
    offsets.push_back(offsets.back() + dist(deterministic_engine()));
  }
  std::vector<char> chars;
  chars.reserve(offsets.back() + 1);
  // Use a pattern so there can be more unique strings in the column
  for (int i = 0; i <= offsets.back(); ++i) { chars.push_back('a' + i % 26); }

  return cudf::make_strings_column(chars, offsets);
}

template <typename T>
std::unique_ptr<cudf::table> create_random_table(cudf::size_type num_columns,
                                                 cudf::size_type col_bytes,
                                                 bool include_validity)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (int idx = 0; idx < num_columns; idx++) {
    columns.emplace_back(create_random_column<T>(col_bytes, include_validity));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

// TODO: create random mixed table
