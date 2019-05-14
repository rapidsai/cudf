#include <algorithm>

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "utilities/nvtx/nvtx_utils.h"
#include "bitmask/legacy_bitmask.hpp"


template<typename T, typename Tout, typename F>
__global__
void gpu_binary_op(const T *lhs_data, const gdf_valid_type *lhs_valid,
                   const T *rhs_data, const gdf_valid_type *rhs_valid,
                   gdf_size_type size, Tout *results, F functor) {
    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int start = tid + blkid * blksz;
    int step = blksz * gridsz;
    if ( lhs_valid || rhs_valid ) {  // has valid mask
        for (int i=start; i<size; i+=step) {
            if (gdf_is_valid(lhs_valid, i) && gdf_is_valid(rhs_valid, i))
                results[i] = functor.apply(lhs_data[i], rhs_data[i]);
        }
    } else {                         // no valid mask
        for (int i=start; i<size; i+=step) {
            results[i] = functor.apply(lhs_data[i], rhs_data[i]);
        }
    }
}

template<typename T, typename Tout, typename F>
struct BinaryOp {
    static
    gdf_error launch(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {

        // Return successully right away for empty inputs
        if((0 == lhs->size) || (0 == rhs->size)){
          return GDF_SUCCESS;
        }

        GDF_REQUIRE(lhs->size == rhs->size, GDF_COLUMN_SIZE_MISMATCH);
        GDF_REQUIRE(lhs->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
        GDF_REQUIRE(lhs->dtype == rhs->dtype, GDF_UNSUPPORTED_DTYPE);

        PUSH_RANGE("LIBGDF_BINARY_OP", BINARY_OP_COLOR);
        // find optimal blocksize
        int mingridsize, blocksize;
        CUDA_TRY(
            cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize,
                                               gpu_binary_op<T, Tout, F>)
        );
        // find needed gridsize
        int neededgridsize = (lhs->size + blocksize - 1) / blocksize;
        int gridsize = std::min(mingridsize, neededgridsize);

        F functor;
        gpu_binary_op<<<gridsize, blocksize>>>(
            // inputs
            (const T*)lhs->data, lhs->valid,
            (const T*)rhs->data, rhs->valid,
            lhs->size,
            // output
            (Tout*)output->data,
            // action
            functor
        );

        cudaDeviceSynchronize();

        POP_RANGE();

        CUDA_CHECK_LAST();
        return GDF_SUCCESS;
    }
};

template<typename T, typename F>
struct ArithOp {
    static
    gdf_error launch(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
        GDF_REQUIRE(output->dtype == lhs->dtype, GDF_UNSUPPORTED_DTYPE);
        return BinaryOp<T, T, F>::launch(lhs, rhs, output);
    }
};

template<typename T, typename F>
struct LogicalOp {
    static
    gdf_error launch(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
        GDF_REQUIRE(output->dtype == GDF_INT8, GDF_UNSUPPORTED_DTYPE);
        return BinaryOp<T, int8_t, F>::launch(lhs, rhs, output);
    }
};


#define DEF_ARITH_OP_REAL(F)                                                  \
gdf_error F##_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) { \
    switch ( lhs->dtype ) {                                                   \
    case GDF_FLOAT32: return F##_f32(lhs, rhs, output);                       \
    case GDF_FLOAT64: return F##_f64(lhs, rhs, output);                       \
    default: return GDF_UNSUPPORTED_DTYPE;                                    \
    }                                                                         \
}

#define DEF_ARITH_OP_NUM(F)                                                   \
gdf_error F##_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) { \
    switch ( lhs->dtype ) {                                                   \
    case GDF_INT32:   return F##_i32(lhs, rhs, output);                       \
    case GDF_INT64:   return F##_i64(lhs, rhs, output);                       \
    case GDF_FLOAT32: return F##_f32(lhs, rhs, output);                       \
    case GDF_FLOAT64: return F##_f64(lhs, rhs, output);                       \
    default: return GDF_UNSUPPORTED_DTYPE;                                    \
    }                                                                         \
}

// Arithmeitc

template<typename T>
struct DeviceAdd {
    __device__
    T apply(T lhs, T rhs) {
        return lhs + rhs;
    }
};

template<typename T>
struct DeviceSub {
    __device__
    T apply(T lhs, T rhs) {
        return lhs - rhs;
    }
};

template<typename T>
struct DeviceMul {
    __device__
    T apply(T lhs, T rhs) {
        return lhs * rhs;
    }
};

template<typename T>
struct DeviceFloorDivInt {
    __device__
    T apply(T lhs, T rhs) {
        return std::floor((double)lhs / (double)rhs);
    }
};

template<typename T>
struct DeviceFloorDivReal {
    __device__
    T apply(T lhs, T rhs) {
        return std::floor(lhs / rhs);
    }
};

template<typename T>
struct DeviceDiv {
    __device__
    T apply(T lhs, T rhs) {
        return lhs / rhs;
    }
};

DEF_ARITH_OP_NUM(gdf_add)

gdf_error gdf_add_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<int32_t, DeviceAdd<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_add_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<int64_t, DeviceAdd<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_add_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<float, DeviceAdd<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_add_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<double, DeviceAdd<double> >::launch(lhs, rhs, output);
}

DEF_ARITH_OP_NUM(gdf_sub)

gdf_error gdf_sub_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<int32_t, DeviceSub<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_sub_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<int64_t, DeviceSub<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_sub_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<float, DeviceSub<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_sub_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<double, DeviceSub<double> >::launch(lhs, rhs, output);
}

DEF_ARITH_OP_NUM(gdf_mul)

gdf_error gdf_mul_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<int32_t, DeviceMul<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_mul_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<int64_t, DeviceMul<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_mul_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<float, DeviceMul<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_mul_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<double, DeviceMul<double> >::launch(lhs, rhs, output);
}

DEF_ARITH_OP_NUM(gdf_floordiv)

gdf_error gdf_floordiv_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<int32_t, DeviceFloorDivInt<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_floordiv_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<int64_t, DeviceFloorDivInt<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_floordiv_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<float, DeviceFloorDivReal<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_floordiv_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<double, DeviceFloorDivReal<double> >::launch(lhs, rhs, output);
}

DEF_ARITH_OP_REAL(gdf_div)

gdf_error gdf_div_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<float, DeviceDiv<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_div_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<double, DeviceDiv<double> >::launch(lhs, rhs, output);
}


// logical


#define DEF_LOGICAL_OP_NUM(F)                                                 \
gdf_error F##_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) { \
    switch ( lhs->dtype ) {                                                   \
    case GDF_INT8:      return F##_i8(lhs, rhs, output);                      \
    case GDF_STRING_CATEGORY:                                                 \
    case GDF_INT32:     return F##_i32(lhs, rhs, output);                     \
    case GDF_INT64:     return F##_i64(lhs, rhs, output);                     \
    case GDF_FLOAT32:   return F##_f32(lhs, rhs, output);                     \
    case GDF_FLOAT64:   return F##_f64(lhs, rhs, output);                     \
    case GDF_DATE32:    return F##_i32(lhs, rhs, output);                     \
    case GDF_DATE64:    return F##_i64(lhs, rhs, output);                     \
    case GDF_TIMESTAMP: return F##_i64(lhs, rhs, output);                     \
    default: return GDF_UNSUPPORTED_DTYPE;                                    \
    }                                                                         \
}

template<typename T>
struct DeviceGt {
    __device__
    bool apply(T lhs, T rhs) {
        return lhs > rhs;
    }
};

template<typename T>
struct DeviceGe {
    __device__
    bool apply(T lhs, T rhs) {
        return lhs >= rhs;
    }
};

template<typename T>
struct DeviceLt {
    __device__
    bool apply(T lhs, T rhs) {
        return lhs < rhs;
    }
};

template<typename T>
struct DeviceLe {
    __device__
    bool apply(T lhs, T rhs) {
        return lhs <= rhs;
    }
};

template<typename T>
struct DeviceEq {
    __device__
    bool apply(T lhs, T rhs) {
        return lhs == rhs;
    }
};


template<typename T>
struct DeviceNe {
    __device__
    bool apply(T lhs, T rhs) {
        return lhs != rhs;
    }
};

DEF_LOGICAL_OP_NUM(gdf_gt)

gdf_error gdf_gt_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int8_t, DeviceGt<int8_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_gt_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int32_t, DeviceGt<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_gt_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int64_t, DeviceGt<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_gt_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<float, DeviceGt<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_gt_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<double, DeviceGt<double> >::launch(lhs, rhs, output);
}

DEF_LOGICAL_OP_NUM(gdf_ge)

gdf_error gdf_ge_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int8_t, DeviceGe<int8_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_ge_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int32_t, DeviceGe<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_ge_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int64_t, DeviceGe<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_ge_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<float, DeviceGe<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_ge_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<double, DeviceGe<double> >::launch(lhs, rhs, output);
}


DEF_LOGICAL_OP_NUM(gdf_lt)

gdf_error gdf_lt_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int8_t, DeviceLt<int8_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_lt_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int32_t, DeviceLt<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_lt_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int64_t, DeviceLt<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_lt_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<float, DeviceLt<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_lt_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<double, DeviceLt<double> >::launch(lhs, rhs, output);
}

DEF_LOGICAL_OP_NUM(gdf_le)

gdf_error gdf_le_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int8_t, DeviceLe<int8_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_le_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int32_t, DeviceLe<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_le_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int64_t, DeviceLe<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_le_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<float, DeviceLe<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_le_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<double, DeviceLe<double> >::launch(lhs, rhs, output);
}

DEF_LOGICAL_OP_NUM(gdf_eq)

gdf_error gdf_eq_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int8_t, DeviceEq<int8_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_eq_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int32_t, DeviceEq<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_eq_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int64_t, DeviceEq<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_eq_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<float, DeviceEq<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_eq_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<double, DeviceEq<double> >::launch(lhs, rhs, output);
}

DEF_LOGICAL_OP_NUM(gdf_ne)

gdf_error gdf_ne_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int8_t, DeviceNe<int8_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_ne_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int32_t, DeviceNe<int32_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_ne_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<int64_t, DeviceNe<int64_t> >::launch(lhs, rhs, output);
}

gdf_error gdf_ne_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<float, DeviceNe<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_ne_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return LogicalOp<double, DeviceNe<double> >::launch(lhs, rhs, output);
}


// bitwise


#define DEF_BITWISE_OP(F)                                                 \
gdf_error F##_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) { \
    switch ( lhs->dtype ) {                                                   \
    case GDF_INT8:    return F##_i8(lhs, rhs, output);                        \
    case GDF_INT32:   return F##_i32(lhs, rhs, output);                       \
    case GDF_INT64:   return F##_i64(lhs, rhs, output);                       \
    default: return GDF_UNSUPPORTED_DTYPE;                                    \
    }                                                                         \
}

#define DEF_BITWISE_IMPL(POSTFIX, CTYPE, TEMPLATE)                                       \
gdf_error gdf_bitwise_##POSTFIX(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {  \
    return ArithOp<CTYPE, TEMPLATE<CTYPE> >::launch(lhs, rhs, output);                   \
}

#define DEF_BITWISE_IMPL_GROUP(NAME, TEMPLATE)      \
DEF_BITWISE_OP(gdf_bitwise_##NAME)                  \
DEF_BITWISE_IMPL(NAME##_i8, int8_t, TEMPLATE)       \
DEF_BITWISE_IMPL(NAME##_i32, int32_t, TEMPLATE)     \
DEF_BITWISE_IMPL(NAME##_i64, int64_t, TEMPLATE)


template<typename T>
struct DeviceBitwiseAnd {
    __device__
    T apply(T lhs, T rhs) {
        return lhs & rhs;
    }
};


template<typename T>
struct DeviceBitwiseOr {
    __device__
    T apply(T lhs, T rhs) {
        return lhs | rhs;
    }
};


template<typename T>
struct DeviceBitwiseXor {
    __device__
    T apply(T lhs, T rhs) {
        return lhs ^ rhs;
    }
};

DEF_BITWISE_IMPL_GROUP(and, DeviceBitwiseAnd)
DEF_BITWISE_IMPL_GROUP(or, DeviceBitwiseOr)
DEF_BITWISE_IMPL_GROUP(xor, DeviceBitwiseXor)


// validity

gdf_column gdf_validity_column(const gdf_column &col) {
    gdf_column ret;
    ret.data = col.valid;
    // TODO: this will need to be changed when gdf_valid_type is changed to 4 byte
    ret.dtype = GDF_INT8;
    ret.valid = nullptr;
    ret.size = (col.size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
    return ret;
}

gdf_error gdf_validity_and(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    GDF_REQUIRE ( output->valid, GDF_VALIDITY_MISSING);
    gdf_column x = gdf_validity_column(*lhs);
    gdf_column y = gdf_validity_column(*rhs);
    gdf_column z = gdf_validity_column(*output);
    
    // TODO: this will need to be changed when gdf_valid_type is changed to 4 byte
    if ( x.data == nullptr && y.data != nullptr ) {
        CUDA_TRY( cudaMemcpy(z.data, y.data, y.size, cudaMemcpyDeviceToDevice) );
        return GDF_SUCCESS;
    } 
    else if ( x.data != nullptr && y.data == nullptr ) {
        CUDA_TRY( cudaMemcpy(z.data, x.data, x.size, cudaMemcpyDeviceToDevice) );
        return GDF_SUCCESS;
    } 
    else if ( x.data == nullptr && y.data == nullptr ) {
        CUDA_TRY( cudaMemset(z.data, 0xff, x.size) );
        return GDF_SUCCESS;
    }
    
    gdf_bitwise_and_i8(&x, &y, &z);
    return GDF_SUCCESS;
}
