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

#include <cudf/column/column_factories.hpp>
#include <cudf/utils/traits.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {

std::unique_ptr<column> make_numeric_column(
    data_type type, size_type size, mask_state state, cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");
}
}  // namespace cudf
