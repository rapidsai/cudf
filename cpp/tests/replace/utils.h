/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include <thrust/device_vector.h>

#include <gdf/gdf.h>

//! traits to get gdf dtype from primitive type
template <class U>
struct TypeTraits {};

#define TYPE_FACTORY(U, D)                                                     \
    template <>                                                                \
    struct TypeTraits<U> {                                                     \
        static constexpr gdf_dtype dtype = GDF_##D;                            \
    }

TYPE_FACTORY(std::int8_t, INT8);
TYPE_FACTORY(std::int16_t, INT16);
TYPE_FACTORY(std::int32_t, INT32);
TYPE_FACTORY(std::int64_t, INT64);
TYPE_FACTORY(float, FLOAT32);
TYPE_FACTORY(double, FLOAT64);

#undef TYPE_FACTORY

//! Convert thrust device vector to gdf_column
template <class T>
static inline gdf_column
MakeGdfColumn(thrust::device_vector<T> &device_vector) {
    return gdf_column{
      .data       = thrust::raw_pointer_cast(device_vector.data()),
      .valid      = nullptr,
      .size       = device_vector.size(),
      .dtype      = TypeTraits<T>::dtype,
      .null_count = 0,
      .dtype_info = {},
    };
}

//! Convert STL vector to gdf_column
template <class T>
static inline gdf_column
MakeGdfColumn(std::vector<T> &vector) {
    return gdf_column{
      .data       = vector.data(),
      .valid      = nullptr,
      .size       = vector.size(),
      .dtype      = TypeTraits<T>::dtype,
      .null_count = 0,
      .dtype_info = {},
    };
}
