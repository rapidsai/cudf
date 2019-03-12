/*
 *
 * Code edits and additions
 * 		Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
 */

#include <cmath>
#include <algorithm>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include <bitmask/legacy_bitmask.hpp>

template<typename T, typename Tout, typename F>
__global__
void gpu_unary_op(const T *data, const gdf_valid_type *valid,
                  gdf_size_type size, Tout *results, F functor) {
    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int start = tid + blkid * blksz;
    int step = blksz * gridsz;
    if ( valid ) {  // has valid mask
        for (int i=start; i<size; i+=step) {
            if ( gdf_is_valid(valid, i) )
                results[i] = functor.apply(data[i]);
        }
    } else {        // no valid mask
        for (int i=start; i<size; i+=step) {
            results[i] = functor.apply(data[i]);
        }
    }
}

template<typename T, typename Tout, typename F>
struct UnaryOp {
    static
    gdf_error launch(gdf_column *input, gdf_column *output) {

        // Return immediately for empty inputs
        if((0==input->size))
        {
          return GDF_SUCCESS;
        }

        /* check for size of the columns */
        if (input->size != output->size) {
            return GDF_COLUMN_SIZE_MISMATCH;
        }

        // find optimal blocksize
        int mingridsize, blocksize;
        CUDA_TRY(
            cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize,
                                               gpu_unary_op<T, Tout, F>)
        );
        // find needed gridsize
        int neededgridsize = (input->size + blocksize - 1) / blocksize;
        int gridsize = std::min(neededgridsize, mingridsize);

        F functor;
        gpu_unary_op<<<gridsize, blocksize>>>(
            // input
            (const T*)input->data, input->valid, input->size,
            // output
            (Tout*)output->data,
            // action
            functor
        );

        CUDA_CHECK_LAST();
        return GDF_SUCCESS;
    }
};


template<typename T, typename F>
struct MathOp {
    static
    gdf_error launch(gdf_column *input, gdf_column *output) {
        return UnaryOp<T, T, F>::launch(input, output);
    }
};


#define DEF_UNARY_OP_REAL(F)                                        \
gdf_error F##_generic(gdf_column *input, gdf_column *output) {      \
    switch ( input->dtype ) {                                       \
    case GDF_FLOAT32: return F##_f32(input, output);                \
    case GDF_FLOAT64: return F##_f64(input, output);                \
    default: return GDF_UNSUPPORTED_DTYPE;                          \
    }                                                               \
}

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

// trig functions

template<typename T>
struct DeviceSin {
    __device__
    T apply(T data) {
        return std::sin(data);
    }
};

template<typename T>
struct DeviceCos {
    __device__
    T apply(T data) {
        return std::cos(data);
    }
};

template<typename T>
struct DeviceTan {
    __device__
    T apply(T data) {
        return std::tan(data);
    }
};

template<typename T>
struct DeviceArcSin {
    __device__
    T apply(T data) {
        return std::asin(data);
    }
};

template<typename T>
struct DeviceArcCos {
    __device__
    T apply(T data) {
        return std::acos(data);
    }
};

template<typename T>
struct DeviceArcTan {
    __device__
    T apply(T data) {
        return std::atan(data);
    }
};

DEF_UNARY_OP_REAL(gdf_sin)

gdf_error gdf_sin_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceSin<float> >::launch(input, output);
}

gdf_error gdf_sin_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceSin<double> >::launch(input, output);
}

DEF_UNARY_OP_REAL(gdf_cos)

gdf_error gdf_cos_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceCos<float> >::launch(input, output);
}

gdf_error gdf_cos_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceCos<double> >::launch(input, output);
}

DEF_UNARY_OP_REAL(gdf_tan)

gdf_error gdf_tan_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceTan<float> >::launch(input, output);
}

gdf_error gdf_tan_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceTan<double> >::launch(input, output);
}

DEF_UNARY_OP_REAL(gdf_asin)

gdf_error gdf_asin_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceArcSin<float> >::launch(input, output);
}

gdf_error gdf_asin_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceArcSin<double> >::launch(input, output);
}

DEF_UNARY_OP_REAL(gdf_acos)

gdf_error gdf_acos_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceArcCos<float> >::launch(input, output);
}

gdf_error gdf_acos_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceArcCos<double> >::launch(input, output);
}

DEF_UNARY_OP_REAL(gdf_atan)

gdf_error gdf_atan_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceArcTan<float> >::launch(input, output);
}

gdf_error gdf_atan_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceArcTan<double> >::launch(input, output);
}

// exponential functions

template<typename T>
struct DeviceExp {
    __device__
    T apply(T data) {
        return std::exp(data);
    }
};

template<typename T>
struct DeviceLog {
    __device__
    T apply(T data) {
        return std::log(data);
    }
};

DEF_UNARY_OP_REAL(gdf_exp)

gdf_error gdf_exp_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceExp<float> >::launch(input, output);
}

gdf_error gdf_exp_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceExp<double> >::launch(input, output);
}

DEF_UNARY_OP_REAL(gdf_log)

gdf_error gdf_log_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceLog<float> >::launch(input, output);
}

gdf_error gdf_log_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceLog<double> >::launch(input, output);
}

// exponential functions

template<typename T>
struct DeviceSqrt {
    __device__
    T apply(T data) {
        return std::sqrt(data);
    }
};

DEF_UNARY_OP_REAL(gdf_sqrt)

gdf_error gdf_sqrt_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceSqrt<float> >::launch(input, output);
}

gdf_error gdf_sqrt_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceSqrt<double> >::launch(input, output);
}

// rounding functions

template<typename T>
struct DeviceCeil {
    __device__
    T apply(T data) {
        return std::ceil(data);
    }
};

template<typename T>
struct DeviceFloor {
    __device__
    T apply(T data) {
        return std::floor(data);
    }
};

DEF_UNARY_OP_REAL(gdf_ceil)

gdf_error gdf_ceil_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceCeil<float> >::launch(input, output);
}

gdf_error gdf_ceil_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceCeil<double> >::launch(input, output);
}

DEF_UNARY_OP_REAL(gdf_floor)

gdf_error gdf_floor_f32(gdf_column *input, gdf_column *output) {
    return MathOp<float, DeviceFloor<float> >::launch(input, output);
}

gdf_error gdf_floor_f64(gdf_column *input, gdf_column *output) {
    return MathOp<double, DeviceFloor<double> >::launch(input, output);
}


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
        thrust::copy(rmm::exec_policy()->on(0), input->valid, input->valid + gdf_num_bitmask_elements(input->size), output->valid);            \
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
        thrust::copy(rmm::exec_policy()->on(0), input->valid, input->valid + gdf_num_bitmask_elements(input->size), output->valid);    \
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
