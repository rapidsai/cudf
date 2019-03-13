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
#include "utilities/wrapper_types.hpp"
#include "utilities/type_dispatcher.hpp"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>


template<typename TypeFrom, typename TypeTo>
struct DeviceCast {
    __device__
    TypeTo apply(TypeFrom data) {
        return static_cast<TypeTo>(cudf::detail::unwrap(data));
    }
};

template<typename TypeFrom, typename TypeTo, int64_t units_factor>
struct UpCasting {
    __device__
    TypeTo apply(TypeFrom data) {
        auto udata = cudf::detail::unwrap(data);
        return static_cast<TypeTo>(udata*units_factor);
    }
};

template<typename TypeFrom, typename TypeTo, int64_t units_factor>
struct DownCasting {
    __device__
    TypeTo apply(TypeFrom data) {
        auto udata = cudf::detail::unwrap(data);
        return static_cast<TypeTo>((udata-(units_factor-1)*(udata<0))/units_factor); //ceiling only when data is negative
    }
};


template <typename TypeFrom>
struct CastTo_Dispatcher {
    template <typename TypeTo>
    gdf_error operator()(gdf_column *input, gdf_column *output) {
        return UnaryOp< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
    }
};


template <typename TypeFrom> // This can be either cudf::date32 or cudf::date64
struct CastDateTo_Dispatcher {
    // Cast to other date type
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_same<TypeTo, cudf::date32>::value || std::is_same<TypeTo, cudf::date64>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        if( input->dtype == GDF_DATE64 && output->dtype == GDF_DATE32 ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 86400000> >::launch(input, output);
        else if( input->dtype == GDF_DATE32 && output->dtype == GDF_DATE64 ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 86400000> >::launch(input, output);
        else // both columns have same type
            return UnaryOp< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
    }
    
    // Cast to cudf::timestamp
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_same<TypeTo, cudf::timestamp>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        if( input->dtype == GDF_DATE32 && ( output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_s ) ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 86400> >::launch(input, output);
        else if( input->dtype == GDF_DATE32 && ( output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_ms ) ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 86400000> >::launch(input, output);
        else if( input->dtype == GDF_DATE32 && ( output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_us ) ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 86400000000> >::launch(input, output);
        else if( input->dtype == GDF_DATE32 && ( output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_ns ) ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 86400000000000> >::launch(input, output);
        else if( input->dtype == GDF_DATE64 && output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_us) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( input->dtype == GDF_DATE64 && output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_s) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( input->dtype == GDF_DATE64 && output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_ms) 
            return UnaryOp< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
        else if( input->dtype == GDF_DATE64 && output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_ns) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000000> >::launch(input, output);
        else
            return GDF_UNSUPPORTED_DTYPE;
    }

    // cast to arithmetic or cudf::category
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_arithmetic<TypeTo>::value || std::is_same<TypeTo, cudf::category>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return UnaryOp< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
    }
};


struct CastTimestampTo_Dispatcher {
    using TypeFrom = cudf::timestamp;

    // cast to date
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_same<TypeTo, cudf::date32>::value || std::is_same<TypeTo, cudf::date64>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_s ) && output->dtype == GDF_DATE32 ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 86400> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ms ) && output->dtype == GDF_DATE32 ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 86400000> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_us ) && output->dtype == GDF_DATE32 ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 86400000000> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ns ) && output->dtype == GDF_DATE32 ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 86400000000000> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_s ) && output->dtype == GDF_DATE64 ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ms ) && output->dtype == GDF_DATE64 ) 
            return UnaryOp< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_us ) && output->dtype == GDF_DATE64 ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ns ) && output->dtype == GDF_DATE64 ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000000> >::launch(input, output);
        else
            return GDF_UNSUPPORTED_DTYPE;
    }

    // cast to another cudf::timestamp
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_same<TypeTo, cudf::timestamp>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        if( input->dtype_info.time_unit == TIME_UNIT_s && output->dtype_info.time_unit == TIME_UNIT_ms ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && output->dtype_info.time_unit == TIME_UNIT_s ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_s && output->dtype_info.time_unit == TIME_UNIT_us ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_us && output->dtype_info.time_unit == TIME_UNIT_s ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_s && output->dtype_info.time_unit == TIME_UNIT_ns ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000000000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && output->dtype_info.time_unit == TIME_UNIT_s ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000000000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_us && output->dtype_info.time_unit == TIME_UNIT_ns ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && output->dtype_info.time_unit == TIME_UNIT_us ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && output->dtype_info.time_unit == TIME_UNIT_ns ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && output->dtype_info.time_unit == TIME_UNIT_ms ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_us && output->dtype_info.time_unit == TIME_UNIT_ms ) 
            return UnaryOp<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && output->dtype_info.time_unit == TIME_UNIT_us ) 
            return UnaryOp<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, 1000> >::launch(input, output);
        else
            return GDF_TIMESTAMP_RESOLUTION_MISMATCH;
    }

    // cast to arithmetic or cudf::category
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_arithmetic<TypeTo>::value || std::is_same<TypeTo, cudf::category>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return UnaryOp< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
    }
};

struct CastFrom_Dispatcher
{
    // Cast from cudf::date32 or cudf::date64
    template <typename TypeFrom>
    typename std::enable_if_t<
        std::is_same<TypeFrom, cudf::date32>::value || std::is_same<TypeFrom, cudf::date64>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return cudf::type_dispatcher(output->dtype,
                                    CastDateTo_Dispatcher<TypeFrom>{},
                                    input, output);
    }
    
    // Cast from cudf::timestamp
    template <typename TypeFrom>
    typename std::enable_if_t<
        std::is_same<TypeFrom, cudf::timestamp>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return cudf::type_dispatcher(output->dtype,
                                    CastTimestampTo_Dispatcher{},
                                    input, output);
    }
    
    // Arithmetic and cudf::category
    template <typename TypeFrom>
    typename std::enable_if_t<
        std::is_arithmetic<TypeFrom>::value || std::is_same<TypeFrom, cudf::category>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return cudf::type_dispatcher(output->dtype,
                                    CastTo_Dispatcher<TypeFrom>{},
                                    input, output);
    }
};


gdf_error gdf_cast(gdf_column *input, gdf_column *output) {
    if (input->valid && output->valid) {
        thrust::copy(rmm::exec_policy()->on(0), input->valid, input->valid + gdf_num_bitmask_elements(input->size), output->valid);
    }

    return cudf::type_dispatcher(input->dtype,
                                CastFrom_Dispatcher{},
                                input, output);
}

gdf_error gdf_cast_generic_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i8_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i32_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i64_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f32_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f64_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date32_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date64_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_timestamp_to_f32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_generic_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i8_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i32_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i64_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f32_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f64_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date32_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date64_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_timestamp_to_f64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_generic_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i8_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i32_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i64_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f32_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f64_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date32_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date64_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_timestamp_to_i8(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_generic_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i8_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i32_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i64_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f32_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f64_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date32_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date64_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_timestamp_to_i32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_generic_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i8_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i32_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i64_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f32_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f64_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date32_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date64_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_timestamp_to_i64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_generic_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i8_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i32_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i64_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f32_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f64_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date32_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date64_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_timestamp_to_date32(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_generic_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i8_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i32_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i64_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f32_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f64_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date32_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date64_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_timestamp_to_date64(gdf_column *input, gdf_column *output) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_generic_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i8_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_i64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_f64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_date64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}
gdf_error gdf_cast_timestamp_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit) {
    return gdf_cast(input, output);
}


