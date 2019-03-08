/*
 *
 * Code edits and additions
 * 		Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
 */

#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "utilities/type_dispatcher.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include "cudf.h"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <algorithm>
#include <type_traits>

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

struct DeviceSin {
    template<typename T>
    __device__
    T apply(T data) {
        return std::sin(data);
    }
};

struct DeviceCos {
    template<typename T>
    __device__
    T apply(T data) {
        return std::cos(data);
    }
};

struct DeviceTan {
    template<typename T>
    __device__
    T apply(T data) {
        return std::tan(data);
    }
};

struct DeviceArcSin {
    template<typename T>
    __device__
    T apply(T data) {
        return std::asin(data);
    }
};

struct DeviceArcCos {
    template<typename T>
    __device__
    T apply(T data) {
        return std::acos(data);
    }
};

struct DeviceArcTan {
    template<typename T>
    __device__
    T apply(T data) {
        return std::atan(data);
    }
};

// exponential functions

struct DeviceExp {
    template<typename T>
    __device__
    T apply(T data) {
        return std::exp(data);
    }
};

struct DeviceLog {
    template<typename T>
    __device__
    T apply(T data) {
        return std::log(data);
    }
};

struct DeviceSqrt {
    template<typename T>
    __device__
    T apply(T data) {
        return std::sqrt(data);
    }
};

// rounding functions

struct DeviceCeil {
    template<typename T>
    __device__
    T apply(T data) {
        return std::ceil(data);
    }
};

struct DeviceFloor {
    template<typename T>
    __device__
    T apply(T data) {
        return std::floor(data);
    }
};

template <typename Op>
struct MathOpDispatcher {
    template <typename T>
    typename std::enable_if_t<std::is_floating_point<T>::value, gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return MathOp<T,Op>::launch(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_floating_point<T>::value, gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};

gdf_error gdf_sin_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceSin>{},
                                input, output);
}

gdf_error gdf_cos_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceCos>{},
                                input, output);
}

gdf_error gdf_tan_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceTan>{},
                                input, output);
}

gdf_error gdf_asin_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceArcSin>{},
                                input, output);
}

gdf_error gdf_acos_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceArcCos>{},
                                input, output);
}

gdf_error gdf_atan_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceArcTan>{},
                                input, output);
}

gdf_error gdf_exp_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceExp>{},
                                input, output);
}

gdf_error gdf_log_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceLog>{},
                                input, output);
}

gdf_error gdf_sqrt_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceSqrt>{},
                                input, output);
}

gdf_error gdf_ceil_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceCeil>{},
                                input, output);
}

gdf_error gdf_floor_generic(gdf_column *input, gdf_column *output) {
    return cudf::type_dispatcher(input->dtype,
                                MathOpDispatcher<DeviceFloor>{},
                                input, output);
}

