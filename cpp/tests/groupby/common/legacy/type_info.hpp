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

#include <cudf/cudf.h>
#include <cudf/groupby.hpp>

namespace cudf {
// forward decls
struct DeviceMin;
struct DeviceMax;
struct DeviceSum;

namespace test {
using namespace cudf::groupby;

/**---------------------------------------------------------------------------*
 * @brief Maps a operators enum value to it's corresponding binary
 * operator functor.
 *
 * @tparam op The enum to map to its corresponding functor
 *---------------------------------------------------------------------------**/
template <operators op> struct corresponding_functor { using type = void; };
template <> struct corresponding_functor<MIN> { using type = DeviceMin; };
template <> struct corresponding_functor<MAX> { using type = DeviceMax; };
template <> struct corresponding_functor<SUM> { using type = DeviceSum; };
template <> struct corresponding_functor<COUNT> { using type = DeviceSum; };
template <operators op>
using corresponding_functor_t = typename corresponding_functor<op>::type;

/**---------------------------------------------------------------------------*
 * @brief Determines accumulator type based on input type and operation.
 *
 * @tparam InputType The type of the input to the aggregation operation
 * @tparam op The aggregation operation performed
 * @tparam dummy Dummy for SFINAE
 *---------------------------------------------------------------------------**/
template <typename SourceType, operators op,
          typename dummy = void>
struct expected_result_type {
  using type = void;
};

// Computing MIN of SourceType, use SourceType accumulator
template <typename SourceType>
struct expected_result_type<SourceType, MIN> {
  using type = SourceType;
};

// Computing MAX of SourceType, use SourceType accumulator
template <typename SourceType>
struct expected_result_type<SourceType, MAX> {
  using type = SourceType;
};

// Always use int64_t accumulator for COUNT
// TODO Use `gdf_size_type`
template <typename SourceType>
struct expected_result_type<SourceType, COUNT> {
  using type = gdf_size_type;
};

// Always use `double` as output of MEAN
template <typename SourceType>
struct expected_result_type<SourceType, MEAN> { using type = double; };

// Summing integers of any type, always use int64_t accumulator
template <typename SourceType>
struct expected_result_type<
    SourceType, SUM, std::enable_if_t<std::is_integral<SourceType>::value>> {
  using type = int64_t;
};

// Summing float/doubles, use same type accumulator
template <typename SourceType>
struct expected_result_type<
    SourceType, SUM,
    std::enable_if_t<std::is_floating_point<SourceType>::value>> {
  using type = SourceType;
};

template <typename SourceType>
struct expected_result_type<
    SourceType, MEDIAN> {
  using type = double;
};
template <typename SourceType>
struct expected_result_type<
    SourceType, QUANTILE> {
  using type = double;
};
template <typename SourceType, operators op>
using expected_result_t = typename expected_result_type<SourceType, op>::type;
}  // namespace test
}  // namespace cudf
