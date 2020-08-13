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

 #include <cudf/lists/list_view.cuh>
 #include <cudf/lists/lists_column_device_view.cuh>

 #include <cstdio>

namespace cudf 
{

__device__ bool list_view::operator == (list_view const& rhs) const
{
    printf("CALEB: list_view::operator ==()!");

    return false;
}

}