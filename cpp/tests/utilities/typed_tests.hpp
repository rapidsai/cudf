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
#include <tests/utilities/type_list.hpp>

#include <array>
#include <tuple>

/**---------------------------------------------------------------------------*
 * @filename typed_tests.hpp
 * @brief Provides centralized abstractions for use in Google Test
 * type-parameterized tests.
 *
 * These abstractions should be used for consistency across tests as well as
 * future-proofing against the addition of any new types in the future.
 *---------------------------------------------------------------------------**/
namespace cudf {
namespace test {
namespace detail {

template <typename TYPES, std::size_t... Indices>
constexpr std::array<cudf::type_id, sizeof...(Indices)> types_to_ids_impl(
    std::index_sequence<Indices...>) {
  return {{cudf::exp::type_to_id<GetType<TYPES, Indices>>()...}};
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

/**---------------------------------------------------------------------------*
 * @brief Provides a list of all numeric types supported in libcudf for use in a
 * GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all numeric types in libcudf
 * TYPED_TEST_CAST(MyTypedFixture, cudf::test::NumericTypes);
 * ```
 *---------------------------------------------------------------------------**/
using NumericTypes =
    cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

/**---------------------------------------------------------------------------*
 * @brief Provides a list of all types supported in libcudf for use in a GTest
 * typed test.
 *
 * @note Currently does not provide any of the "wrapped" types, e.g., timestamp,
 * category, etc.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all types supported by libcudf
 * TYPED_TEST_CAST(MyTypedFixture, cudf::test::AllTypes);
 * ```
 *---------------------------------------------------------------------------**/
using AllTypes = Concat<NumericTypes>;

/**---------------------------------------------------------------------------*
 * @brief `std::array` of of all `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 *---------------------------------------------------------------------------**/
static constexpr auto all_type_ids{detail::types_to_ids<AllTypes>()};

/**---------------------------------------------------------------------------*
 * @brief `std::array` of of all numeric `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 *---------------------------------------------------------------------------**/
static constexpr auto numeric_type_ids{detail::types_to_ids<NumericTypes>()};

/**---------------------------------------------------------------------------*
 * @brief `std::array` of of all non-numeric `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 *---------------------------------------------------------------------------**/
static constexpr std::array<cudf::type_id, 6> non_numeric_type_ids{
    cudf::EMPTY,     cudf::BOOL8,    cudf::DATE32,
    cudf::TIMESTAMP, cudf::CATEGORY, cudf::STRING};

//static_assert(cudf::type_id::NUM_TYPE_IDS == all_type_ids.size(),
//              "Mismatch in number of types");

}  // namespace test
}  // namespace cudf