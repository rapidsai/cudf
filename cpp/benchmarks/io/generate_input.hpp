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

// TODO limit to integral? SFINAE might already do it
template <typename T>
T random_elem()
{
  static constexpr T lower_bound = std::numeric_limits<T>::lowest();
  static constexpr T upper_bound = std::numeric_limits<T>::max();

  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::normal_distribution<> gaussian{0., stddev<T>()};
  auto elem = gaussian(engine);
  if (lower_bound >= 0) elem = abs(elem);
  elem = std::max(std::min(elem, (double)upper_bound), (double)lower_bound);
  std::cout << T(elem) << ' ';
  return T(elem);
}

template <typename T>
std::unique_ptr<cudf::table> create_random_table(cudf::size_type num_columns,
                                                 cudf::size_type num_rows,
                                                 bool include_validity)
{
  // 1% of elements are invalid
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 100 == 0 ? false : true; });
  std::vector<cudf::test::fixed_width_column_wrapper<T>> src_cols(num_columns);
  for (int idx = 0; idx < num_columns; idx++) {
    auto rand_elements =
      cudf::test::make_counting_transform_iterator(0, [](T i) { return random_elem<T>(); });
    if (include_validity) {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows, valids);
    } else {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows);
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
  std::transform(src_cols.begin(),
                 src_cols.end(),
                 columns.begin(),
                 [](cudf::test::fixed_width_column_wrapper<T>& in) {
                   auto ret = in.release();
                   ret->has_nulls();
                   return ret;
                 });
  return std::make_unique<cudf::table>(std::move(columns));
}