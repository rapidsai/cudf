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

#ifndef COMPILED_BINARY_OPS_H
#define COMPILED_BINARY_OPS_H

#include <cudf/types.hpp>
#include <cudf/binaryop.hpp>

namespace cudf {
namespace binops {
namespace compiled {

gdf_error binary_operation(gdf_column* out,
                           gdf_column* lhs,
                           gdf_column* rhs,
                           gdf_binary_operator ope);

} // namespace compiled
} // namespace binops
} // namespace cudf


#endif // COMPILED_BINARY_OPS_H
