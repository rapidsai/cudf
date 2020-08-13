/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
// #include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cstdio>
#include <cuda_runtime.h>

/**
 * @file list_view.cuh
 * @brief Class definition for cudf::list_view.
 */

namespace cudf {

/**
 * @brief A non-owning, immutable view of device data that represents
 * a list of elements of arbitrary type (including further nested lists).
 *
 */
class list_view {

    public:

        // Not intended to be invoked. This allows type_dispatcher to function.
        __device__ bool operator == (list_view const& rhs) const
        {
            release_assert(!"list_view::operator ==() should not be called directly.");
            printf("CALEB: list_view::operator ==()!\n");

            return false;
        }

};

}  // namespace cudf
