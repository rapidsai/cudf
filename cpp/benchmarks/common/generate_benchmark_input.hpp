#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <memory>
#include <random>
#include <vector>

/**
 * @file generate_benchmark_input.hpp
 * @brief Contains functions that generate columns filled with random data.
 *
 * Also includes utilies that generate random tables.
 *
 * The distribution of random data is meant to simulate real-world data. For example, numerical
 * values are generated using a normal distribution with a zero mean. Therefore, different column
 * types are filled using different distributions. The distributions are documented in the
 * functions where they are used.
 *
 * Currently, the data generation is done on the CPU and the data is then copied to the device
 * memory.
 */

/**
 * @brief Mersenne Twister engine with static seed.
 *
 * Produces the same random sequence on each run.
 */
auto& deterministic_engine()
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  return engine;
}

/**
 * @brief Type trait for cudf timestamp types.
 */
template <typename>
struct is_timestamp {
  constexpr static bool value = false;
};

template <typename T>
struct is_timestamp<cudf::detail::timestamp<T>> {
  constexpr static bool value = true;
};

/**
 * @brief nanosecond count in the unit of @ref T.
 *
 * @tparam T Timestamp type
 */
template <typename T>
constexpr int64_t nanoseconds()
{
  using ratio = std::ratio_divide<typename T::period, typename cudf::timestamp_ns::period>;
  return ratio::num / ratio::den;
}

/**
 * @brief Creates an random timestamp
 *
 * Generates 'recent' timestamps. All timstamps are earlier that June 2020. The period between the
 * timestamps and June 2020 has a geometric distribution. Most timestamps are within a few years
 * before 2020.
 *
 * @return The random timestamp
 * @tparam T Timestamp type
 */
template <typename T, std::enable_if_t<is_timestamp<T>::value, int> = 0>
T random_element()
{
  // Timestamp for June 2020
  static constexpr int64_t current_ns    = 1591053936l * nanoseconds<cudf::timestamp_s>();
  static constexpr auto timestamp_spread = 1. / (2 * 365 * 24 * 60 * 60);  // one in two years

  // Generate a number of seconds that is 50% likely to be shorter than two years
  static std::geometric_distribution<int64_t> seconds_gen{timestamp_spread};
  // Generate a random value for the nanoseconds within a second
  static std::uniform_int_distribution<int64_t> nanoseconds_gen{0,
                                                                nanoseconds<cudf::timestamp_s>()};

  // Subtract the seconds from the 2020 timestamp to generate a reccent timestamp
  auto const timestamp_ns = current_ns -
                            seconds_gen(deterministic_engine()) * nanoseconds<cudf::timestamp_s>() -
                            nanoseconds_gen(deterministic_engine());
  // Return value in the type's precision
  return T(timestamp_ns / nanoseconds<T>());
}

/**
 * @brief Standard deviation for the Normal distribution used to generate numeric elements.
 *
 * Deviation depends on the type width; wider types -> larger value range.
 */
template <typename T>
constexpr auto stddev()
{
  return 1l << (sizeof(T) * 4);
}

/**
 * @brief Creates an random numeric value with a normal distribution
 *
 * Zero is always used as the mean for teh distribution. Unsigned types are generated as the
 * absolute value of the normal distribution output.
 * Different standard deviations are used depending on the type size, in order to generate larger
 * range of values for when the types supports it.
 *
 * @return The random number
 * @tparam T Numeric type
 */
template <typename T, std::enable_if_t<not is_timestamp<T>::value, int> = 0>
T random_element()
{
  static constexpr T lower_bound = std::numeric_limits<T>::lowest();
  static constexpr T upper_bound = std::numeric_limits<T>::max();

  // Use the type dependent standard deviation
  static std::normal_distribution<> gaussian{0., stddev<T>()};

  auto elem = gaussian(deterministic_engine());
  // Use absolute value for unsigned types
  if (lower_bound >= 0) elem = abs(elem);
  elem = std::max(std::min(elem, (double)upper_bound), (double)lower_bound);

  return T(elem);
}

/**
 * @brief Creates an boolean value with 50:50 probability
 *
 * @return The random boolean value
 */
template <>
bool random_element<bool>()
{
  static std::uniform_int_distribution<> uniform{0, 1};
  return uniform(deterministic_engine()) == 1;
}

/**
 * @brief Creates a column with random content of the given type
 *
 * The templated implementation is used for all fixed width types. String columns are generated
 * using the specialization implemented below.
 *
 * @param[in] col_bytes Size of the generated column, in bytes
 * @param[in] include_validity Whether to include the null mask in the columns
 *
 * @return Column filled with random data
 */
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

/**
 * @brief Creates a string column with random content
 *
 * Uses a Poisson distribution around the mean string length. The average length of elements is 16
 * and currently there is no way to modify this via parameters.
 *
 * Due to random generation of the length of the columns elements, the resulting column will have a
 * slightly different size from @ref col_bytes.
 *
 * @param[in] col_bytes Size of the generated column, in bytes
 * @param[in] include_validity Whether to include the null mask in the columns
 *
 * @return Column filled with random data
 */
template <>
std::unique_ptr<cudf::column> create_random_column<std::string>(cudf::size_type col_bytes,
                                                                bool include_validity)
{
  // TODO: have some elements be null?

  static constexpr int avg_string_len = 16;
  const cudf::size_type num_rows      = col_bytes / avg_string_len;

  std::poisson_distribution<> dist(avg_string_len);

  std::vector<int32_t> offsets{0};
  offsets.reserve(num_rows + 1);
  for (int i = 1; i < num_rows; ++i) {
    offsets.push_back(offsets.back() + dist(deterministic_engine()));
  }
  auto const char_cnt = offsets.back() + 1;
  std::vector<char> chars;
  chars.reserve(char_cnt);
  // Use a pattern so there can be more unique strings in the column
  std::generate_n(std::back_inserter(chars), char_cnt, []() {
    static size_t i = 0;
    return 'a' + (i++ % 26);
  });

  return cudf::make_strings_column(chars, offsets);
}

/**
 * @brief Creates a table with random content of the given type
 *
 * @param[in] num_columns Number of columns in the table
 * @param[in] col_bytes Size of each column, in bytes
 * @param[in] include_validity Whether to include the null mask in the columns
 *
 * @return Table filled with random data
 */
template <typename T>
std::unique_ptr<cudf::table> create_random_table(cudf::size_type num_columns,
                                                 cudf::size_type col_bytes,
                                                 bool include_validity)
{
  return std::make_unique<cudf::table>([&]() {
    std::vector<std::unique_ptr<cudf::column>> columns;
    std::generate_n(std::back_inserter(columns), num_columns, [&]() {
      return create_random_column<T>(col_bytes, include_validity);
    });
    return columns;
  }());
}

// TODO: create random mixed table
