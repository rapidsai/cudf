/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

namespace CUDF_EXPORT cudf {
namespace io::detail {
/**
 * @brief Whether writer writes in chunks or all at once
 */
enum class single_write_mode : bool { YES, NO };

template <typename T>
constexpr bool is_convertible_to_string_column()
{
  // Note: the case (not std::is_same_v<T, bool>)
  // is already covered by is_integral
  return std::is_same_v<T, cudf::string_view> || std::is_integral_v<T> ||
         std::is_floating_point_v<T> || cudf::is_fixed_point<T>() || cudf::is_timestamp<T>() ||
         cudf::is_duration<T>();
}

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
