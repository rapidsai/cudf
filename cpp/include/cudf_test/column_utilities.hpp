/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf::test {

/**
 * @brief Verbosity level of output from column and table comparison functions.
 */
enum class debug_output_level {
  FIRST_ERROR = 0,  // print first error only
  ALL_ERRORS,       // print all errors
  QUIET             // no debug output
};

constexpr size_type default_ulp = 4;

namespace detail {

/**
 * @brief Verifies the property equality of two columns.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL`
 * instead.
 *
 * @param lhs The first column
 * @param rhs The second column
 * @param verbosity Level of debug output verbosity
 *
 * @returns True if the column properties are equal, false otherwise
 */
bool expect_column_properties_equal(cudf::column_view const& lhs,
                                    cudf::column_view const& rhs,
                                    debug_output_level verbosity = debug_output_level::FIRST_ERROR);

/**
 * @brief Verifies the property equivalence of two columns.
 *
 * If the columns don't have nulls, then the nullability equality is relaxed.
 * i.e. the two columns are considered equivalent even if one has a null mask
 * and the other doesn't.
 *
 * @note This function should not be used directly. Use
 * `CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT` instead.
 *
 * @param lhs The first column
 * @param rhs The second column
 * @param verbosity Level of debug output verbosity
 *
 * @returns True if the column properties are equivalent, false otherwise
 */
bool expect_column_properties_equivalent(
  cudf::column_view const& lhs,
  cudf::column_view const& rhs,
  debug_output_level verbosity = debug_output_level::FIRST_ERROR);

/**
 * @brief Verifies the element-wise equality of two columns.
 *
 * Treats null elements as equivalent.
 *
 * @note This function should not be used directly. Use
 * `CUDF_TEST_EXPECT_COLUMNS_EQUAL` instead.
 *
 * @param lhs The first column
 * @param rhs The second column
 * @param verbosity Level of debug output verbosity
 *
 * @returns True if the columns (and their properties) are equal, false otherwise
 */
bool expect_columns_equal(cudf::column_view const& lhs,
                          cudf::column_view const& rhs,
                          debug_output_level verbosity = debug_output_level::FIRST_ERROR);

/**
 * @brief Verifies the element-wise equivalence of two columns.
 *
 * Uses machine epsilon to compare floating point types.
 * Treats null elements as equivalent.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT`
 * instead.
 *
 * @param lhs The first column
 * @param rhs The second column
 * @param verbosity Level of debug output verbosity
 * @param fp_ulps # of ulps of tolerance to allow when comparing
 * floating point values
 *
 * @returns True if the columns (and their properties) are equivalent, false otherwise
 */
bool expect_columns_equivalent(cudf::column_view const& lhs,
                               cudf::column_view const& rhs,
                               debug_output_level verbosity = debug_output_level::FIRST_ERROR,
                               size_type fp_ulps            = cudf::test::default_ulp);

/**
 * @brief Verifies the bitwise equality of two device memory buffers.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_EQUAL_BUFFERS` instead.
 *
 * @param lhs The first buffer
 * @param rhs The second buffer
 * @param size_bytes The number of bytes to check for equality
 */
void expect_equal_buffers(void const* lhs, void const* rhs, std::size_t size_bytes);

}  // namespace detail

/**
 * @brief Verifies the given column is empty
 *
 * @param col The column to check
 */
void expect_column_empty(cudf::column_view const& col);

/**
 * @brief Copy the null bitmask from a column view to a host vector
 *
 * @param c      The column view
 * @returns      Vector of bitmask_type elements
 */
std::vector<bitmask_type> bitmask_to_host(cudf::column_view const& c);

/**
 * @brief Validates bitmask situated in host as per `number_of_elements`
 *
 * This takes care of padded bits
 *
 * @param        expected_mask A vector representing expected mask
 * @param        got_mask A vector representing mask obtained from column
 * @param        number_of_elements number of elements the mask represent
 *
 * @returns      true if both vector match till the `number_of_elements`
 */
bool validate_host_masks(std::vector<bitmask_type> const& expected_mask,
                         std::vector<bitmask_type> const& got_mask_begin,
                         size_type number_of_elements);

/**
 * @brief Copies the data and bitmask of a `column_view` to the host.
 *
 * @tparam T The data type of the elements of the `column_view`
 * @param c the `column_view` to copy from
 * @return std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> first is the
 *  `column_view`'s data, and second is the column's bitmask.
 */
template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> to_host(column_view c)
{
  thrust::host_vector<T> host_data(c.size());
  CUDF_CUDA_TRY(cudaMemcpy(host_data.data(), c.data<T>(), c.size() * sizeof(T), cudaMemcpyDefault));
  return {host_data, bitmask_to_host(c)};
}

/**
 * @brief Copies the data and bitmask of a `column_view` to the host.
 *
 * This is the specialization for `fixed_point` that performs construction of a `fixed_point` from
 * the underlying `rep` type that is stored on the device.
 *
 * @tparam T The data type of the elements of the `column_view`
 * @param c the `column_view` to copy from
 * @return std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> first is the
 *  `column_view`'s data, and second is the column's bitmask.
 */
template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> to_host(column_view c)
{
  using namespace numeric;
  using Rep = typename T::rep;

  auto host_rep_types = thrust::host_vector<Rep>(c.size());

  CUDF_CUDA_TRY(
    cudaMemcpy(host_rep_types.data(), c.begin<Rep>(), c.size() * sizeof(Rep), cudaMemcpyDefault));

  auto to_fp = [&](Rep val) { return T{scaled_integer<Rep>{val, scale_type{c.type().scale()}}}; };
  auto begin = thrust::make_transform_iterator(std::cbegin(host_rep_types), to_fp);
  auto const host_fixed_points = thrust::host_vector<T>(begin, begin + c.size());

  return {host_fixed_points, bitmask_to_host(c)};
}

/**
 * @brief Copies the data and bitmask of a `column_view` of strings
 * column to the host.
 *
 * @throw cudf::logic_error if c is not strings column.
 *
 * @param c the `column_view` of strings to copy from
 * @return std::pair first is `std::vector` of `std::string`
 * and second is the column's bitmask.
 */
template <>
inline std::pair<thrust::host_vector<std::string>, std::vector<bitmask_type>> to_host(column_view c)
{
  thrust::host_vector<std::string> host_data(c.size());
  if (c.size() > c.null_count()) {
    auto const scv     = strings_column_view(c);
    auto const h_chars = cudf::detail::make_std_vector_sync<char>(
      cudf::device_span<char const>(scv.chars().data<char>(), scv.chars().size()),
      cudf::get_default_stream());
    auto const h_offsets = cudf::detail::make_std_vector_sync(
      cudf::device_span<cudf::size_type const>(scv.offsets().data<cudf::size_type>() + scv.offset(),
                                               scv.size() + 1),
      cudf::get_default_stream());

    // build std::string vector from chars and offsets
    std::transform(
      std::begin(h_offsets),
      std::end(h_offsets) - 1,
      std::begin(h_offsets) + 1,
      host_data.begin(),
      [&](auto start, auto end) { return std::string(h_chars.data() + start, end - start); });
  }
  return {std::move(host_data), bitmask_to_host(c)};
}

}  // namespace cudf::test

// Macros for showing line of failure.
#define CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(lhs, rhs)        \
  do {                                                            \
    SCOPED_TRACE(" <--  line of failure\n");                      \
    cudf::test::detail::expect_column_properties_equal(lhs, rhs); \
  } while (0)

#define CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(lhs, rhs)        \
  do {                                                                 \
    SCOPED_TRACE(" <--  line of failure\n");                           \
    cudf::test::detail::expect_column_properties_equivalent(lhs, rhs); \
  } while (0)

#define CUDF_TEST_EXPECT_COLUMNS_EQUAL(lhs, rhs...)     \
  do {                                                  \
    SCOPED_TRACE(" <--  line of failure\n");            \
    cudf::test::detail::expect_columns_equal(lhs, rhs); \
  } while (0)

#define CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lhs, rhs...)     \
  do {                                                       \
    SCOPED_TRACE(" <--  line of failure\n");                 \
    cudf::test::detail::expect_columns_equivalent(lhs, rhs); \
  } while (0)

#define CUDF_TEST_EXPECT_EQUAL_BUFFERS(lhs, rhs, size_bytes)        \
  do {                                                              \
    SCOPED_TRACE(" <--  line of failure\n");                        \
    cudf::test::detail::expect_equal_buffers(lhs, rhs, size_bytes); \
  } while (0)
