/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

/*
The sort-based approach is adapted from https://github.com/moderngpu/moderngpu
which has the following license:

> Copyright (c) 2016, Sean Baxter
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this
>    list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
> ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
> ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
> ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> The views and conclusions contained in the software and documentation are those
> of the authors and should not be interpreted as representing official policies,
> either expressed or implied, of the FreeBSD Project.
*/


#include <gdf/gdf.h>
#include <gdf/errorutils.h>

#include <memory>
#include <iostream>

#include "joining.h"

using namespace mgpu;

template <typename T>
void dump_mem(const char name[], const mem_t<T> & mem) {

    auto data = from_mem(mem);
    std::cout << name << " = " ;
    for (int i=0; i < data.size(); ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << "\n";
}

gdf_join_result_type* cffi_wrap(join_result_base *obj) {
    return reinterpret_cast<gdf_join_result_type*>(obj);
}

join_result_base* cffi_unwrap(gdf_join_result_type* hdl) {
    return reinterpret_cast<join_result_base*>(hdl);
}

gdf_error gdf_join_result_free(gdf_join_result_type *result) {
    delete cffi_unwrap(result);
    CUDA_CHECK_LAST();
    return GDF_SUCCESS;
}

void* gdf_join_result_data(gdf_join_result_type *result) {
    return cffi_unwrap(result)->data();
}

size_t gdf_join_result_size(gdf_join_result_type *result) {
    return cffi_unwrap(result)->size();
}


// Size limit due to use of int32 as join output.
// FIXME: upgrade to 64-bit
#define MAX_JOIN_SIZE (0xffffffffu)

#define DEF_JOIN(Fn, T, Joiner)                                             \
gdf_error gdf_##Fn(gdf_column *leftcol, gdf_column *rightcol,               \
                   gdf_join_result_type **out_result) {                     \
    using namespace mgpu;                                                   \
    if ( leftcol->dtype != rightcol->dtype) return GDF_UNSUPPORTED_DTYPE;   \
    if ( leftcol->size >= MAX_JOIN_SIZE ) return GDF_COLUMN_SIZE_TOO_BIG;   \
    if ( rightcol->size >= MAX_JOIN_SIZE ) return GDF_COLUMN_SIZE_TOO_BIG;  \
    std::unique_ptr<join_result<int> > result_ptr(new join_result<int>);    \
    result_ptr->result = Joiner((T*)leftcol->data, leftcol->size,           \
                                (T*)rightcol->data, rightcol->size,         \
                                less_t<T>(), result_ptr->context);          \
    CUDA_CHECK_LAST();                                                      \
    *out_result = cffi_wrap(result_ptr.release());                          \
    return GDF_SUCCESS;                                                     \
}


#define DEF_JOIN_DISP(Fn)                                                   \
gdf_error gdf_##Fn##_generic(gdf_column *leftcol, gdf_column * rightcol,    \
                                 gdf_join_result_type **out_result) {       \
    switch ( leftcol->dtype ){                                              \
    case GDF_INT8:  return gdf_##Fn##_i8(leftcol, rightcol, out_result);    \
    case GDF_INT32: return gdf_##Fn##_i32(leftcol, rightcol, out_result);   \
    case GDF_INT64: return gdf_##Fn##_i64(leftcol, rightcol, out_result);   \
    case GDF_FLOAT32: return gdf_##Fn##_f32(leftcol, rightcol, out_result); \
    case GDF_FLOAT64: return gdf_##Fn##_f64(leftcol, rightcol, out_result); \
    default: return GDF_UNSUPPORTED_DTYPE;                                  \
    }                                                                       \
}

#define DEF_INNER_JOIN(Fn, T) DEF_JOIN(inner_join_ ## Fn, T, inner_join)
#define DEF_INNER_JOIN_HASH(Fn, T) DEF_JOIN(inner_join_ ## Fn, T, inner_join_hash)
DEF_JOIN_DISP(inner_join)
DEF_INNER_JOIN_HASH(i8,  int8_t)
DEF_INNER_JOIN_HASH(i16, int16_t)
DEF_INNER_JOIN_HASH(i32, int32_t)
DEF_INNER_JOIN_HASH(i64, int64_t)
DEF_INNER_JOIN(f32, float)
DEF_INNER_JOIN(f64, double)


#define DEF_LEFT_JOIN(Fn, T) DEF_JOIN(left_join_ ## Fn, T, left_join)
DEF_JOIN_DISP(left_join)
DEF_LEFT_JOIN(i8,  int8_t)
DEF_LEFT_JOIN(i32, int32_t)
DEF_LEFT_JOIN(i64, int64_t)
DEF_LEFT_JOIN(f32, float)
DEF_LEFT_JOIN(f64, double)


#define DEF_OUTER_JOIN(Fn, T) DEF_JOIN(outer_join_ ## Fn, T, outer_join)
DEF_JOIN_DISP(outer_join)
DEF_OUTER_JOIN(i8,  int8_t)
DEF_OUTER_JOIN(i32, int32_t)
DEF_OUTER_JOIN(i64, int64_t)
DEF_OUTER_JOIN(f32, float)
DEF_OUTER_JOIN(f64, double)

