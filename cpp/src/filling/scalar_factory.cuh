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
 
namespace cudf {

namespace detail {

struct scalar_factory {
    gdf_scalar value;
  
    template <typename T>
    struct scalar {
      T value;
      bool is_valid;
  
      __device__
      T data(gdf_index_type index) { return value; }
  
      __device__
      bool valid(gdf_index_type index) { return is_valid; }
    };
  
    template <typename T>
    scalar<T> make() {
      T val{}; // Safe type pun, compiler should optimize away the memcpy
      memcpy(&val, &value.data, sizeof(T));
      return scalar<T>{val, value.is_valid};
    }
};

} // namespace detail

} // namespace cudf
