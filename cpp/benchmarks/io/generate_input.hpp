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
  return 1l << (sizeof(T) * 4);  // wider distribution for wider types
}

template <typename>
struct is_timestamp {
  constexpr static bool value = false;
};

template <typename T>
struct is_timestamp<cudf::detail::timestamp<T>> {
  constexpr static bool value = true;
};

template <typename Ts>
constexpr int64_t nanoseconds();

template <>
constexpr int64_t nanoseconds<cudf::timestamp_D>()
{
  return 24l * 60 * 60 * 1000 * 1000 * 1000;
}

template <>
constexpr int64_t nanoseconds<cudf::timestamp_s>()
{
  return 1000l * 1000 * 1000;
}

template <>
constexpr int64_t nanoseconds<cudf::timestamp_ms>()
{
  return 1000l * 1000;
}

template <>
constexpr int64_t nanoseconds<cudf::timestamp_us>()
{
  return 1000l;
}

template <>
constexpr int64_t nanoseconds<cudf::timestamp_ns>()
{
  return 1;
}

template <typename T, std::enable_if_t<is_timestamp<T>::value, int> = 0>
T random_element()
{
  static constexpr int64_t current_ns    = 1591053936l * nanoseconds<cudf::timestamp_s>();
  static constexpr auto timestamp_spread = 1. / (2 * 365 * 24 * 60 * 60);  // one in two years

  // TODO: extract seed + engine into a single function
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::geometric_distribution<int64_t> seconds_gen{timestamp_spread};
  static std::uniform_int_distribution<int64_t> nanoseconds_gen{0,
                                                                nanoseconds<cudf::timestamp_s>()};

  // most timestamps will corespond to recent years
  auto const timestamp_ns =
    current_ns - seconds_gen(engine) * nanoseconds<cudf::timestamp_s>() - nanoseconds_gen(engine);

  return T(timestamp_ns / nanoseconds<T>());  // convert to the type precision
}

template <typename T, std::enable_if_t<not is_timestamp<T>::value, int> = 0>
T random_element()
{
  static constexpr T lower_bound = std::numeric_limits<T>::lowest();
  static constexpr T upper_bound = std::numeric_limits<T>::max();

  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::normal_distribution<> gaussian{0., stddev<T>()};

  auto elem = gaussian(engine);
  if (lower_bound >= 0) elem = abs(elem);
  elem = std::max(std::min(elem, (double)upper_bound), (double)lower_bound);

  return T(elem);
}

template <>
bool random_element<bool>()
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<> uniform{0, 1};
  return uniform(engine) == 1;
}

template <typename T>
std::unique_ptr<cudf::column> create_random_column(cudf::size_type col_bytes, bool include_validity)
{
  const cudf::size_type num_rows = col_bytes / sizeof(T);

  // every 100th element is invalid
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

  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  std::poisson_distribution<> dist(avg_string_len);

  std::vector<int32_t> offsets{0};
  for (int i = 1; i < num_rows; ++i) { offsets.push_back(offsets.back() + dist(engine)); }
  std::vector<char> chars;
  chars.reserve(offsets.back() + 1);
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