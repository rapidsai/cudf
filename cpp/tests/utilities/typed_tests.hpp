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

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <gtest/gtest.h>

#include <array>
#include <tuple>

namespace cudf {
namespace test {

namespace detail {

template <typename Tuple, std::size_t... Indices>
constexpr std::array<cudf::type_id, sizeof...(Indices)> to_array_impl(
    std::index_sequence<Indices...>) {
  return {{cudf::exp::type_to_id<std::tuple_element_t<Indices, Tuple>>()...}};
}

/**---------------------------------------------------------------------------*
 * @brief Converts a tuple of type `Ts...` into a `std::array` of the
 * corresponding `cudf::type_id`s for each `T` in `Ts...`
 *
 * Example:
 * ```
 * auto array = tuple_to_data_type_array<std::tuple<int32_t, float>>();
 * array == {INT32, FLOAT};
 * ```
 *
 * @tparam Tuple tuple of types `Ts...`
 * @return `std::array` of `type_id`s corresponding to each type in `Ts...`
 *---------------------------------------------------------------------------**/
template <typename Tuple>
constexpr auto types_to_ids() {
  constexpr std::size_t N =
      std::tuple_size<std::remove_reference_t<Tuple>>::value;
  return to_array_impl<Tuple>(std::make_index_sequence<N>());
}

template <typename... Ts>
struct tuple_to_test_types_impl {};

template <typename... Ts>
struct tuple_to_test_types_impl<std::tuple<Ts...>> {
  using types = ::testing::Types<Ts...>;
};

/**---------------------------------------------------------------------------*
 * @brief Converts a `std::tuple<Ts...>` into a Google Test
 * `testing::Types<Ts...>` for use in GTest typed tests.
 *
 * @tparam Tuple The tuple whose constituent types will be used in the GTest
 * typed tests
 *---------------------------------------------------------------------------**/
template <typename Tuple>
using tuple_to_test_types = typename tuple_to_test_types_impl<Tuple>::types;

using AllTypesTuple =
    std::tuple<int8_t, int16_t, int32_t, int64_t, float, double>;

using NumericTypesTuple =
    std::tuple<int8_t, int16_t, int32_t, int64_t, float, double>;
}  // namespace detail

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
using AllTypes = detail::tuple_to_test_types<detail::AllTypesTuple>;

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
using NumericTypes = detail::tuple_to_test_types<detail::NumericTypesTuple>;

/**---------------------------------------------------------------------------*
 * @brief `std::array` of of all numeric `cudf::type_id`s
 *---------------------------------------------------------------------------**/
static constexpr auto numeric_data_types{
    detail::types_to_ids<detail::NumericTypesTuple>()};

/**---------------------------------------------------------------------------*
 * @brief `std::array` of of all non-numeric `cudf::type_id`s
 *---------------------------------------------------------------------------**/
static constexpr auto non_numeric_data_types{
    cudf::data_type{cudf::EMPTY},    cudf::data_type{cudf::BOOL8},
    cudf::data_type{cudf::DATE32},   cudf::data_type{cudf::TIMESTAMP},
    cudf::data_type{cudf::CATEGORY}, cudf::data_type{cudf::STRING}};

static_assert(cudf::type_id::NUM_TYPE_IDS ==
                  (non_numeric_data_types.size() + numeric_data_types.size()),
              "Mismatch in number of types");
}  // namespace test
}  // namespace cudf