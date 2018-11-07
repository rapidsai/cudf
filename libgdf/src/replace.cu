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

#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/replace.h>

#include <gdf/gdf.h>

namespace {

//! traits to get primitive type from gdf dtype
template <gdf_dtype DTYPE>
struct gdf_dtype_traits {};

#define DTYPE_FACTORY(DTYPE, T)                                                \
    template <>                                                                \
    struct gdf_dtype_traits<GDF_##DTYPE> {                                     \
        typedef T value_type;                                                  \
    }

DTYPE_FACTORY(INT8, std::int8_t);
DTYPE_FACTORY(INT16, std::int16_t);
DTYPE_FACTORY(INT32, std::int32_t);
DTYPE_FACTORY(INT64, std::int64_t);
DTYPE_FACTORY(FLOAT32, float);
DTYPE_FACTORY(FLOAT64, double);
DTYPE_FACTORY(DATE32, std::int32_t);
DTYPE_FACTORY(DATE64, std::int64_t);
DTYPE_FACTORY(TIMESTAMP, std::int64_t);

#undef DTYPE_FACTORY

/// /brief Replace kernel
/// \param[in/out] data with elements to be replaced
/// \param[in] values contains the replacement values
/// \param[in] to_replace_begin begin pointer of `to_replace` array
/// \param[in] to_replace_begin end pointer of `to_replace` array
template <class T>
__global__ void
replace_kernel(T *const                          data,
               const std::size_t                 data_size,
               const T *const                    values,
               const thrust::device_ptr<const T> to_replace_begin,
               const thrust::device_ptr<const T> to_replace_end) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < data_size;
         i += blockDim.x * gridDim.x) {
        // TODO: find by map kernel
        const thrust::device_ptr<const T> found_ptr = thrust::find(
          thrust::device, to_replace_begin, to_replace_end, data[i]);

        if (found_ptr != to_replace_end) {
            typename thrust::iterator_traits<
              const thrust::device_ptr<const T>>::difference_type
              value_found_index = thrust::distance(to_replace_begin, found_ptr);

            data[i] = values[value_found_index];
        }
    }
}

/// /brief Call replace kernel according to primitive type T
/// \param[in/out] data with elements to be replaced
/// \param[in] data_size number of elements in data
/// \param[in] to_replace contains values that will be replaced
/// \param[in] values contains the replacement values
/// \param[in] replacement_ptrdiff to get the end pointer of `to_replace` array
template <class T>
static inline gdf_error
Replace(T *const             data,
        const std::size_t    data_size,
        const T *const       to_replace,
        const T *const       values,
        const std::ptrdiff_t replacement_ptrdiff) {
    const std::size_t blocks = std::ceil(data_size / 256.);

    const thrust::device_ptr<const T> to_replace_begin(to_replace);
    const thrust::device_ptr<const T> to_replace_end(to_replace_begin
                                                     + replacement_ptrdiff);

    replace_kernel<T><<<blocks, 256>>>(  // TODO: calc blocks and threads
      data,
      data_size,
      values,
      to_replace_begin,
      to_replace_end);

    return GDF_SUCCESS;
}

/// \brief Check if two gdf_columns have the same size
/// \param[in] to_replace is a gdf_column
/// \param[in] values is a gdf_column
static inline bool
NotEqualReplacementSize(const gdf_column *to_replace,
                        const gdf_column *values) {
    return to_replace->size != values->size;
}

/// \brief Check if the three gdf columns have the same dtype
/// \param[in] column is as gdf_column
/// \param[in] to_replace is a gdf_column
/// \param[in] values is a gdf_column
static inline bool
NotSameDType(const gdf_column *column,
             const gdf_column *to_replace,
             const gdf_column *values) {
    return column->dtype != to_replace->dtype
           || to_replace->dtype != values->dtype;
}

}  // namespace

/// \brief For each value in `to_replace`, find all instances of that value
///        in `column` and replace it with the corresponding value in `values`.
/// \param[in/out] column data
/// \param[in] to_replace contains values of column that will be replaced
/// \param[in] values contains the replacement values
///
/// Note that `to_replace` and `values` are related by the index
gdf_error
gdf_find_and_replace_all(gdf_column *      column,
                         const gdf_column *to_replace,
                         const gdf_column *values) {
    if (NotEqualReplacementSize(to_replace, values)) {
        return GDF_COLUMN_SIZE_MISMATCH;
    }

    if (NotSameDType(column, to_replace, values)) { return GDF_CUDA_ERROR; }

    switch (column->dtype) {
#define REPLACE_CASE(DTYPE)                                                    \
    case GDF_##DTYPE: {                                                        \
        using value_type = gdf_dtype_traits<GDF_##DTYPE>::value_type;          \
        return Replace(static_cast<value_type *>(column->data),                \
                       static_cast<std::size_t>(column->size),                 \
                       static_cast<value_type *>(to_replace->data),            \
                       static_cast<value_type *>(values->data),                \
                       static_cast<std::ptrdiff_t>(values->size));             \
    }

        REPLACE_CASE(INT8);
        REPLACE_CASE(INT16);
        REPLACE_CASE(INT32);
        REPLACE_CASE(INT64);
        REPLACE_CASE(FLOAT32);
        REPLACE_CASE(FLOAT64);
        REPLACE_CASE(DATE32);
        REPLACE_CASE(DATE64);
        REPLACE_CASE(TIMESTAMP);

#undef REPLACE_CASE

    case GDF_invalid:
    default: return GDF_UNSUPPORTED_DTYPE;
    }
}
