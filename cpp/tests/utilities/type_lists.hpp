/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <cudf/wrappers/bool.hpp>

#include <array>
#include <tuple>

/**---------------------------------------------------------------------------*
 * @filename type_lists.hpp
 * @brief Provides centralized type lists for use in Google Test
 * type-parameterized tests.
 *
 * These lists should be used for consistency across tests as well as
 * future-proofing against the addition of any new types in the future.
 *---------------------------------------------------------------------------**/
namespace cudf {
namespace test {
namespace detail {

template <typename TYPES, std::size_t... Indices>
constexpr std::array<cudf::type_id, sizeof...(Indices)> types_to_ids_impl(
    std::index_sequence<Indices...>) {
  return {{cudf::experimental::type_to_id<GetType<TYPES, Indices>>()...}};
}

/**---------------------------------------------------------------------------*
 * @brief Converts a `Types` list of types into a `std::array` of the
 * corresponding `cudf::type_id`s for each type in the list
 *
 * Example:
 * ```
 * auto array = types_to_ids<Types<int32_t, float>>();
 * array == {INT32, FLOAT};
 * ```
 *
 * @tparam TYPES List of types to conver to `type_id`s
 * @return `std::array` of `type_id`s corresponding to each type in `TYPES`
 *---------------------------------------------------------------------------**/
template <typename TYPES>
constexpr auto types_to_ids() {
  constexpr auto N = GetSize<TYPES>;
  return types_to_ids_impl<TYPES>(std::make_index_sequence<N>());
}
}  // namespace detail

/**
 * @brief Provides a list of all floating point types supported in libcudf for
 * use in a GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all floating point types in libcudf
 * TYPED_TEST_CASE(MyTypedFixture, cudf::test::FloatingPointTypes);
 * ```
 */
using FloatingPointTypes = cudf::test::Types<float, double>;

/**---------------------------------------------------------------------------*
 * @brief Provides a list of all numeric types supported in libcudf for use in a
 * GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all numeric types in libcudf
 * TYPED_TEST_CASE(MyTypedFixture, cudf::test::NumericTypes);
 * ```
 *---------------------------------------------------------------------------**/
using NumericTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float,
                                       double, cudf::experimental::bool8>;

/**---------------------------------------------------------------------------*
 * @brief Provides a list of all timestamp types supported in libcudf for use
 * in a GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all timestamp types in libcudf
 * TYPED_TEST_CASE(MyTypedFixture, cudf::test::TimestampTypes);
 * ```
 *---------------------------------------------------------------------------**/
using TimestampTypes = cudf::test::Types<timestamp_D, timestamp_s, timestamp_ms,
                                         timestamp_us, timestamp_ns>;

/**---------------------------------------------------------------------------*
 * @brief Provides a list of all string types supported in libcudf for use in a
 * GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all string types in libcudf
 * TYPED_TEST_CASE(MyTypedFixture, cudf::test::StringTypes);
 * ```
 *---------------------------------------------------------------------------**/
using StringTypes = cudf::test::Types<string_view>;

/**---------------------------------------------------------------------------*
 * @brief Provides a list of all fixed-width element types for use in GTest
 * typed tests.
 * 
 * Example:
 * ```
 * // Invokes all typed fixture tests for all fixed-width types in libcudf
 * TYPED_TEST_CASE(MyTypedFixture, cudf::test::FixedWidthTypes);
 * ```
 *---------------------------------------------------------------------------**/
using FixedWidthTypes = Concat<NumericTypes, TimestampTypes>;

/**---------------------------------------------------------------------------*
 * @brief Provides a list of sortable types for use in GTest typed tests.
 * 
 * Example:
 * ```
 * // Invokes all typed fixture tests for all sortable types in libcudf
 * TYPED_TEST_CASE(MyTypedFixture, cudf::test::ComparableTypes);
 * ```
 *---------------------------------------------------------------------------**/
using ComparableTypes = Concat<NumericTypes, TimestampTypes, StringTypes>;

/**---------------------------------------------------------------------------*
 * @brief Provides a list of all types supported in libcudf for use in a GTest
 * typed test.
 *
 * @note Currently does not provide any of the "wrapped" types, e.g.,
 * category, etc.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all types supported by libcudf
 * TYPED_TEST_CASE(MyTypedFixture, cudf::test::AllTypes);
 * ```
 *---------------------------------------------------------------------------**/
using AllTypes = Concat<NumericTypes, TimestampTypes>;

/**---------------------------------------------------------------------------*
 * @brief `std::array` of all `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 *---------------------------------------------------------------------------**/
static constexpr auto all_type_ids{detail::types_to_ids<AllTypes>()};

/**---------------------------------------------------------------------------*
 * @brief `std::array` of all numeric `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 *---------------------------------------------------------------------------**/
static constexpr auto numeric_type_ids{detail::types_to_ids<NumericTypes>()};

/**---------------------------------------------------------------------------*
 * @brief `std::array` of all timestamp `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 *---------------------------------------------------------------------------**/
static constexpr std::array<cudf::type_id, 5> timestamp_type_ids{
    detail::types_to_ids<TimestampTypes>()};

/**---------------------------------------------------------------------------*
 * @brief `std::array` of all non-numeric `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 *---------------------------------------------------------------------------**/
static constexpr std::array<cudf::type_id, 8> non_numeric_type_ids{
    cudf::EMPTY,
    cudf::TIMESTAMP_DAYS,
    cudf::TIMESTAMP_SECONDS,
    cudf::TIMESTAMP_MILLISECONDS,
    cudf::TIMESTAMP_MICROSECONDS,
    cudf::TIMESTAMP_NANOSECONDS,
    cudf::CATEGORY,
    cudf::STRING};

/**---------------------------------------------------------------------------*
 * @brief `std::array` of all non-fixed-width `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 *---------------------------------------------------------------------------**/
static constexpr std::array<cudf::type_id, 3> non_fixed_width_type_ids{
    cudf::EMPTY,    
    cudf::CATEGORY,
    cudf::STRING};

}  // namespace test
}  // namespace cudf
