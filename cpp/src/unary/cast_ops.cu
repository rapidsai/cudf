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

/*
 *
 * Code edits and additions
 * 		Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
 */

#include "unary_ops.cuh"

#include "rmm/thrust_rmm_allocator.h"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

// casting

template<typename From, typename To>
struct DeviceCast {
    __device__
    To apply(From data) {
        return (To)data;
    }
};

template<typename From, typename To, int64_t units_factor>
struct UpCasting {
    __device__
    To apply(From data) {
        return (To)(data*units_factor);
    }
};

template<typename From, typename To, int64_t units_factor>
struct DownCasting {
    __device__
    To apply(From data) {
        return (To)((data-(units_factor-1)*(data<0))/units_factor); //ceiling only when data is negative
    }
};

#define DEF_CAST_OP(TO)                                                       \
gdf_error gdf_cast_generic_to_##TO(gdf_column *input, gdf_column *output) {   \
    switch ( input->dtype ) {                                                 \
    case      GDF_INT8: return gdf_cast_i8_to_##TO(input, output);            \
    case     GDF_INT32: return gdf_cast_i32_to_##TO(input, output);           \
    case     GDF_INT64: return gdf_cast_i64_to_##TO(input, output);           \
    case   GDF_FLOAT32: return gdf_cast_f32_to_##TO(input, output);           \
    case   GDF_FLOAT64: return gdf_cast_f64_to_##TO(input, output);           \
    case    GDF_DATE32: return gdf_cast_date32_to_##TO(input, output);        \
    case    GDF_DATE64: return gdf_cast_date64_to_##TO(input, output);        \
    case GDF_TIMESTAMP: return gdf_cast_timestamp_to_##TO(input, output);     \
    default: return GDF_UNSUPPORTED_DTYPE;                                    \
    }                                                                         \
}
#define DEF_CAST_OP_TS(TO)                                                                          \
gdf_error gdf_cast_generic_to_##TO(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {\
    switch ( input->dtype ) {                                                                       \
    case      GDF_INT8: return gdf_cast_i8_to_##TO(input, output, time_unit);                       \
    case     GDF_INT32: return gdf_cast_i32_to_##TO(input, output, time_unit);                      \
    case     GDF_INT64: return gdf_cast_i64_to_##TO(input, output, time_unit);                      \
    case   GDF_FLOAT32: return gdf_cast_f32_to_##TO(input, output, time_unit);                      \
    case   GDF_FLOAT64: return gdf_cast_f64_to_##TO(input, output, time_unit);                      \
    case    GDF_DATE32: return gdf_cast_date32_to_##TO(input, output, time_unit);                   \
    case    GDF_DATE64: return gdf_cast_date64_to_##TO(input, output, time_unit);                   \
    case GDF_TIMESTAMP: return gdf_cast_timestamp_to_##TO(input, output, time_unit);                \
    default: return GDF_UNSUPPORTED_DTYPE;                                                          \
    }                                                                                               \
}

// Castings are differentiate between physical and logical ones.
// In physical casting only change the physical representation, for example from GDF_FLOAT32 (float) to GDF_FLOAT64 (double)
// on the other hand, casting between date timestamps needs also perform some calculations according to the time unit:
// - when the source or destination datatype is GDF_DATE32, the value is multiplied or divided by the amount of timeunits by day
// - when datatypes are timestamps, the value is multiplied or divided according to the S.I. nano 10^-9, micro 10^-6, milli 10^-3
// No calculation is necessary when casting between GDF_DATE64 and GDF_TIMESTAMP (with ms as time unit), because are logically and physically the same thing

#define DEF_CAST_IMPL(VFROM, VTO, TFROM, TTO, LTFROM, LTO)                                                      \
gdf_error gdf_cast_##VFROM##_to_##VTO(gdf_column *input, gdf_column *output) {                                  \
    GDF_REQUIRE(input->dtype == LTFROM, GDF_UNSUPPORTED_DTYPE);                                                 \
                                                                                                                \
                                                                                                                \
    output->dtype = LTO;                                                                                        \
    if (input->valid && output->valid) {                                                                        \
        gdf_size_type num_chars_bitmask = gdf_get_num_chars_bitmask(input->size);                               \
        thrust::copy(rmm::exec_policy()->on(0), input->valid, input->valid + num_chars_bitmask, output->valid);            \
    }                                                                                                           \
                                                                                                                \
    /* Handling datetime logical castings */                                                                    \
    if( LTFROM == GDF_DATE64 && LTO == GDF_DATE32 )                                                             \
        return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 86400000> >::launch(input, output);                  \
    else if( LTFROM == GDF_DATE32 && LTO == GDF_DATE64 )                                                        \
        return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 86400000> >::launch(input, output);                    \
    else if( ( LTFROM == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_s ) && LTO == GDF_DATE32 )   \
        return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 86400> >::launch(input, output);                     \
    else if( ( LTFROM == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ms ) && LTO == GDF_DATE32 )  \
        return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 86400000> >::launch(input, output);                  \
    else if( ( LTFROM == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_us ) && LTO == GDF_DATE32 )  \
        return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 86400000000> >::launch(input, output);               \
    else if( ( LTFROM == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ns ) && LTO == GDF_DATE32 )  \
        return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 86400000000000> >::launch(input, output);            \
    else if( ( LTFROM == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_s ) && LTO == GDF_DATE64 )   \
        return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000> >::launch(input, output);                        \
    else if( ( LTFROM == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_us ) && LTO == GDF_DATE64 )  \
        return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000> >::launch(input, output);                      \
    else if( ( LTFROM == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ns ) && LTO == GDF_DATE64 )  \
        return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000000> >::launch(input, output);                   \
    /* Handling only physical castings */                                                                       \
    return UnaryOp<TFROM, TTO, DeviceCast<TFROM, TTO> >::launch(input, output);                                 \
}

// Castings functions where Timestamp is the destination type
#define DEF_CAST_IMPL_TS(VFROM, VTO, TFROM, TTO, LTFROM, LTO)                                           \
gdf_error gdf_cast_##VFROM##_to_##VTO(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) { \
    GDF_REQUIRE(input->dtype == LTFROM, GDF_UNSUPPORTED_DTYPE);                                         \
                                                                                                        \
                                                                                                        \
    output->dtype = LTO;                                                                                \
    output->dtype_info.time_unit = time_unit;                                                           \
    if (input->valid && output->valid) {                                                                \
        gdf_size_type num_chars_bitmask = gdf_get_num_chars_bitmask(input->size);                       \
        thrust::copy(rmm::exec_policy()->on(0), input->valid, input->valid + num_chars_bitmask, output->valid);    \
    }                                                                                                   \
                                                                                                        \
    /* Handling datetime logical castings */                                                            \
    if( LTFROM == GDF_DATE32 && ( LTO == GDF_TIMESTAMP && time_unit == TIME_UNIT_s ) )                  \
        return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 86400> >::launch(input, output);               \
    else if( LTFROM == GDF_DATE32 && ( LTO == GDF_TIMESTAMP && time_unit == TIME_UNIT_ms ) )            \
        return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 86400000> >::launch(input, output);            \
    else if( LTFROM == GDF_DATE32 && ( LTO == GDF_TIMESTAMP && time_unit == TIME_UNIT_us ) )            \
        return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 86400000000> >::launch(input, output);         \
    else if( LTFROM == GDF_DATE32 && ( LTO == GDF_TIMESTAMP && time_unit == TIME_UNIT_ns ) )            \
        return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 86400000000000> >::launch(input, output);      \
    else if( LTFROM == GDF_DATE64 && LTO == GDF_TIMESTAMP && time_unit == TIME_UNIT_us)                 \
        return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000> >::launch(input, output);                \
    else if( LTFROM == GDF_DATE64 && LTO == GDF_TIMESTAMP && time_unit == TIME_UNIT_s)                  \
        return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000> >::launch(input, output);              \
    else if( LTFROM == GDF_DATE64 && LTO == GDF_TIMESTAMP && time_unit == TIME_UNIT_ns)                 \
        return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000000> >::launch(input, output);             \
    else if( LTFROM == GDF_TIMESTAMP && LTO == GDF_TIMESTAMP )                                          \
    {                                                                                                   \
        if( input->dtype_info.time_unit == TIME_UNIT_s && time_unit == TIME_UNIT_ms )                   \
            return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000> >::launch(input, output);            \
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && time_unit == TIME_UNIT_s )              \
            return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000> >::launch(input, output);          \
        else if( input->dtype_info.time_unit == TIME_UNIT_s && time_unit == TIME_UNIT_us )              \
            return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000000> >::launch(input, output);         \
        else if( input->dtype_info.time_unit == TIME_UNIT_us && time_unit == TIME_UNIT_s )              \
            return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000000> >::launch(input, output);       \
        else if( input->dtype_info.time_unit == TIME_UNIT_s && time_unit == TIME_UNIT_ns )              \
            return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000000000> >::launch(input, output);      \
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && time_unit == TIME_UNIT_s )              \
            return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000000000> >::launch(input, output);    \
        else if( input->dtype_info.time_unit == TIME_UNIT_us && time_unit == TIME_UNIT_ns )             \
            return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000> >::launch(input, output);            \
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && time_unit == TIME_UNIT_us )             \
            return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000> >::launch(input, output);          \
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && time_unit == TIME_UNIT_ns )             \
            return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000000> >::launch(input, output);         \
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && time_unit == TIME_UNIT_ms )             \
            return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000000> >::launch(input, output);       \
        else if( input->dtype_info.time_unit == TIME_UNIT_us && time_unit == TIME_UNIT_ms )             \
            return UnaryOp<TFROM, TTO, DownCasting<TFROM, TTO, 1000> >::launch(input, output);          \
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && time_unit == TIME_UNIT_us )             \
            return UnaryOp<TFROM, TTO, UpCasting<TFROM, TTO, 1000> >::launch(input, output);            \
    }                                                                                                   \
    /* Handling only physical castings */                                                               \
    return UnaryOp<TFROM, TTO, DeviceCast<TFROM, TTO> >::launch(input, output);                         \
}

#define DEF_CAST_IMPL_TEMPLATE(ABREV, PHYSICAL_TYPE, LOGICAL_TYPE)                    \
DEF_CAST_OP(ABREV)                                                                    \
DEF_CAST_IMPL(i8,        ABREV,  int8_t, PHYSICAL_TYPE, GDF_INT8,       LOGICAL_TYPE) \
DEF_CAST_IMPL(i32,       ABREV, int32_t, PHYSICAL_TYPE, GDF_INT32,      LOGICAL_TYPE) \
DEF_CAST_IMPL(i64,       ABREV, int64_t, PHYSICAL_TYPE, GDF_INT64,      LOGICAL_TYPE) \
DEF_CAST_IMPL(f32,       ABREV,   float, PHYSICAL_TYPE, GDF_FLOAT32,    LOGICAL_TYPE) \
DEF_CAST_IMPL(f64,       ABREV,  double, PHYSICAL_TYPE, GDF_FLOAT64,    LOGICAL_TYPE) \
DEF_CAST_IMPL(date32,    ABREV, int32_t, PHYSICAL_TYPE, GDF_DATE32,     LOGICAL_TYPE) \
DEF_CAST_IMPL(date64,    ABREV, int64_t, PHYSICAL_TYPE, GDF_DATE64,     LOGICAL_TYPE) \
DEF_CAST_IMPL(timestamp, ABREV, int64_t, PHYSICAL_TYPE, GDF_TIMESTAMP,  LOGICAL_TYPE)

#define DEF_CAST_IMPL_TEMPLATE_TS(ABREV, PHYSICAL_TYPE, LOGICAL_TYPE)                    \
DEF_CAST_OP_TS(ABREV)                                                                    \
DEF_CAST_IMPL_TS(i8,        ABREV,  int8_t, PHYSICAL_TYPE, GDF_INT8,       LOGICAL_TYPE) \
DEF_CAST_IMPL_TS(i32,       ABREV, int32_t, PHYSICAL_TYPE, GDF_INT32,      LOGICAL_TYPE) \
DEF_CAST_IMPL_TS(i64,       ABREV, int64_t, PHYSICAL_TYPE, GDF_INT64,      LOGICAL_TYPE) \
DEF_CAST_IMPL_TS(f32,       ABREV,   float, PHYSICAL_TYPE, GDF_FLOAT32,    LOGICAL_TYPE) \
DEF_CAST_IMPL_TS(f64,       ABREV,  double, PHYSICAL_TYPE, GDF_FLOAT64,    LOGICAL_TYPE) \
DEF_CAST_IMPL_TS(date32,    ABREV, int32_t, PHYSICAL_TYPE, GDF_DATE32,     LOGICAL_TYPE) \
DEF_CAST_IMPL_TS(date64,    ABREV, int64_t, PHYSICAL_TYPE, GDF_DATE64,     LOGICAL_TYPE) \
DEF_CAST_IMPL_TS(timestamp, ABREV, int64_t, PHYSICAL_TYPE, GDF_TIMESTAMP,  LOGICAL_TYPE)

DEF_CAST_IMPL_TEMPLATE(f32, float, GDF_FLOAT32)
DEF_CAST_IMPL_TEMPLATE(f64, double, GDF_FLOAT64)
DEF_CAST_IMPL_TEMPLATE(i8, int8_t, GDF_INT8)
DEF_CAST_IMPL_TEMPLATE(i32, int32_t, GDF_INT32)
DEF_CAST_IMPL_TEMPLATE(i64, int64_t, GDF_INT64)
DEF_CAST_IMPL_TEMPLATE(date32, int32_t, GDF_DATE32)
DEF_CAST_IMPL_TEMPLATE(date64, int64_t, GDF_DATE64)
DEF_CAST_IMPL_TEMPLATE_TS(timestamp, int64_t, GDF_TIMESTAMP)

