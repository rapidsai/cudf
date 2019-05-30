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

#include <algorithm>

#include "launcher.cuh"
#include "utilities/cudf_utils.h"
#include <binaryop.hpp>

namespace cudf {
namespace binops {
namespace compiled {

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


gdf_error gdf_div_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<float, DeviceDiv<float> >::launch(lhs, rhs, output);
}

gdf_error gdf_div_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return ArithOp<double, DeviceDiv<double> >::launch(lhs, rhs, output);
}


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

DEF_ARITH_OP_NUM(gdf_add)
DEF_ARITH_OP_NUM(gdf_sub)
DEF_ARITH_OP_NUM(gdf_mul)
DEF_ARITH_OP_NUM(gdf_floordiv)
DEF_ARITH_OP_REAL(gdf_div)

// logical

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

DEF_LOGICAL_OP_NUM(gdf_gt)
DEF_LOGICAL_OP_NUM(gdf_ge)
DEF_LOGICAL_OP_NUM(gdf_lt)
DEF_LOGICAL_OP_NUM(gdf_le)
DEF_LOGICAL_OP_NUM(gdf_eq)
DEF_LOGICAL_OP_NUM(gdf_ne)


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
DEF_BITWISE_IMPL(NAME##_i8, int8_t, TEMPLATE)       \
DEF_BITWISE_IMPL(NAME##_i32, int32_t, TEMPLATE)     \
DEF_BITWISE_IMPL(NAME##_i64, int64_t, TEMPLATE)     \
DEF_BITWISE_OP(gdf_bitwise_##NAME)


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


gdf_error binary_operation(gdf_column* out,
                           gdf_column* lhs,
                           gdf_column* rhs,
                           gdf_binary_operator ope)
{
    switch (ope)
    {
    case GDF_ADD:
        return gdf_add_generic(lhs, rhs, out);
    case GDF_SUB:
        return gdf_sub_generic(lhs, rhs, out);
    case GDF_MUL:
        return gdf_mul_generic(lhs, rhs, out);
    case GDF_DIV:
        return gdf_floordiv_generic(lhs, rhs, out);
    case GDF_FLOOR_DIV:
        return gdf_div_generic(lhs, rhs, out);
    case GDF_EQUAL:
        return gdf_gt_generic(lhs, rhs, out);
    case GDF_NOT_EQUAL:
        return gdf_ge_generic(lhs, rhs, out);
    case GDF_LESS:
        return gdf_lt_generic(lhs, rhs, out);
    case GDF_GREATER:
        return gdf_le_generic(lhs, rhs, out);
    case GDF_LESS_EQUAL:
        return gdf_eq_generic(lhs, rhs, out);
    case GDF_GREATER_EQUAL:
        return gdf_ne_generic(lhs, rhs, out);
    case GDF_BITWISE_AND:
        return gdf_bitwise_and_generic(lhs, rhs, out);
    case GDF_BITWISE_OR:
        return gdf_bitwise_or_generic(lhs, rhs, out);
    case GDF_BITWISE_XOR:
        return gdf_bitwise_xor_generic(lhs, rhs, out);
    
    default:
        return GDF_INVALID_API_CALL;
    }
}

} // namespace compiled
} // namespace binops
} // namespace cudf
