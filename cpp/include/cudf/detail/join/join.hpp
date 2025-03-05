/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <limits>

namespace CUDF_EXPORT cudf {
namespace detail {

constexpr int DEFAULT_JOIN_CG_SIZE = 2;

constexpr size_type JoinNoneValue = std::numeric_limits<size_type>::min();

enum class join_kind { INNER_JOIN, LEFT_JOIN, FULL_JOIN, LEFT_SEMI_JOIN, LEFT_ANTI_JOIN };

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
