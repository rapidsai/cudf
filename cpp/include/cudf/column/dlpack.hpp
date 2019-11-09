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

#include <cudf/column/column.hpp>

#include <dlpack/dlpack.h>

namespace cudf {

/**
 * @brief Convert a DLPack DLTensor into cudf::columns
 * 
 * @throw TODO
 * 
 * @param tensor a 1D or 2D column-major (Fortran order) tensor
 * @param mr Optional resource to use for device memory allocation
 * 
 * @return Columns converted from DLPack
 */
std::vector<column> from_dlpack(
    DLManagedTensor const& tensor,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Convert cudf::columns into a DLPack DLTensor
 * 
 * @throw TODO
 * 
 * @param columns Columns to convert to DLPack
 * 
 * @return 1D or 2D DLPack tensor (single or multiple columns)
 */
DLManagedTensor to_dlpack(std::vector<column_view> const& columns);

}  // namespace cudf
