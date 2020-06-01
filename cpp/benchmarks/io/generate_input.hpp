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

template <typename T>
T random_numeric()
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

template <typename T>
std::unique_ptr<cudf::column> create_random_column(cudf::size_type col_bytes, bool include_validity)
{
  const cudf::size_type num_rows = col_bytes / sizeof(T);

  // every 100th element is invalid
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 100 == 0 ? false : true; });

  cudf::test::fixed_width_column_wrapper<T> wrapped_col;
  auto rand_elements =
    cudf::test::make_counting_transform_iterator(0, [](auto row) { return random_numeric<T>(); });
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

  std::cout << std::endl;

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