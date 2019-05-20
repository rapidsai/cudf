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

#ifndef _TYPE_INFO_HPP
#define _TYPE_INFO_HPP

#include <groupby.hpp>
#include <utilities/device_atomics.cuh>

#include <algorithm>

namespace cudf {
namespace groupby {
namespace hash {
/**---------------------------------------------------------------------------*
 * @brief Maps a operators enum value to it's corresponding binary
 * operator functor.
 *
 * @tparam op The enum to map to its corresponding functor
 *---------------------------------------------------------------------------**/
template <operators op>
struct corresponding_functor {
  using type = void;
};

template <>
struct corresponding_functor<MIN> {
  using type = DeviceMin;
};

template <>
struct corresponding_functor<MAX> {
  using type = DeviceMax;
};

template <>
struct corresponding_functor<SUM> {
  using type = DeviceSum;
};

template <>
struct corresponding_functor<COUNT> {
  using type = DeviceSum;
};

template <operators op>
using corresponding_functor_t = typename corresponding_functor<op>::type;
/**---------------------------------------------------------------------------*
 * @brief Determines accumulator type based on input type and operation.
 *
 * @tparam InputType The type of the input to the aggregation operation
 * @tparam op The aggregation operation performed
 * @tparam dummy Dummy for SFINAE
 *---------------------------------------------------------------------------**/
template <typename SourceType, operators op, typename dummy = void>
struct target_type {
  using type = void;
};

// Computing MIN of SourceType, use SourceType accumulator
template <typename SourceType>
struct target_type<SourceType, MIN> {
  using type = SourceType;
};

// Computing MAX of SourceType, use SourceType accumulator
template <typename SourceType>
struct target_type<SourceType, MAX> {
  using type = SourceType;
};

// Always use int64_t accumulator for COUNT
template <typename SourceType>
struct target_type<SourceType, COUNT> {
  // TODO Use `gdf_size_type`
  using type = int64_t;
};

// Summing integers of any type, always use int64_t accumulator
template <typename SourceType>
struct target_type<SourceType, SUM,
                   std::enable_if_t<std::is_integral<SourceType>::value>> {
  using type = int64_t;
};

// Summing float/doubles, use same type accumulator
template <typename SourceType>
struct target_type<
    SourceType, SUM,
    std::enable_if_t<std::is_floating_point<SourceType>::value>> {
  using type = SourceType;
};

template <typename SourceType, operators op>
using target_type_t = typename target_type<SourceType, op>::type;


}  // namespace hash
}  // namespace groupby
}  // namespace cudf

#endif