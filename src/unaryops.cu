#include <cmath>

#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>

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
        int gridsize = (input->size + blocksize - 1) / blocksize;

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
    case    GDF_INT8: return gdf_cast_i8_to_##TO(input, output);              \
    case   GDF_INT32: return gdf_cast_i32_to_##TO(input, output);             \
    case   GDF_INT64: return gdf_cast_i64_to_##TO(input, output);             \
    case GDF_FLOAT32: return gdf_cast_f32_to_##TO(input, output);             \
    case GDF_FLOAT64: return gdf_cast_f64_to_##TO(input, output);             \
    default: return GDF_UNSUPPORTED_DTYPE;                                    \
    }                                                                         \
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

#define DEF_CAST_IMPL(VFROM, VTO, TFROM, TTO)                                 \
gdf_error gdf_cast_##VFROM##_to_##VTO(gdf_column *input, gdf_column *output)  \
{ return UnaryOp<TFROM, TTO, DeviceCast<TFROM, TTO> >::launch(input, output); }


DEF_CAST_OP(f32)
DEF_CAST_IMPL(i8,  f32,  int8_t, float)
DEF_CAST_IMPL(i32, f32, int32_t, float)
DEF_CAST_IMPL(i64, f32, int64_t, float)
DEF_CAST_IMPL(f32, f32,   float, float)
DEF_CAST_IMPL(f64, f32,  double, float)

DEF_CAST_OP(f64)
DEF_CAST_IMPL(i8,  f64,  int8_t, double)
DEF_CAST_IMPL(i32, f64, int32_t, double)
DEF_CAST_IMPL(i64, f64, int64_t, double)
DEF_CAST_IMPL(f32, f64,   float, double)
DEF_CAST_IMPL(f64, f64,  double, double)

DEF_CAST_OP(i8)
DEF_CAST_IMPL(i8,  i8,  int8_t, int8_t)
DEF_CAST_IMPL(i32, i8, int32_t, int8_t)
DEF_CAST_IMPL(i64, i8, int64_t, int8_t)
DEF_CAST_IMPL(f32, i8,   float, int8_t)
DEF_CAST_IMPL(f64, i8,  double, int8_t)

DEF_CAST_OP(i32)
DEF_CAST_IMPL(i8,  i32,  int8_t, int32_t)
DEF_CAST_IMPL(i32, i32, int32_t, int32_t)
DEF_CAST_IMPL(i64, i32, int64_t, int32_t)
DEF_CAST_IMPL(f32, i32,   float, int32_t)
DEF_CAST_IMPL(f64, i32,  double, int32_t)

DEF_CAST_OP(i64)
DEF_CAST_IMPL(i8,  i64,  int8_t, int64_t)
DEF_CAST_IMPL(i32, i64, int32_t, int64_t)
DEF_CAST_IMPL(i64, i64, int64_t, int64_t)
DEF_CAST_IMPL(f32, i64,   float, int64_t)
DEF_CAST_IMPL(f64, i64,  double, int64_t)
