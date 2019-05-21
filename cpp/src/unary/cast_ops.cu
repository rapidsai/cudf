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


static constexpr int64_t METRIC_FACTOR = 1000;
static constexpr int64_t METRIC_FACTOR_SQ = METRIC_FACTOR * METRIC_FACTOR;
static constexpr int64_t METRIC_FACTOR_CUBE = METRIC_FACTOR_SQ * METRIC_FACTOR;

static constexpr int64_t SECONDS_IN_DAY = 86400;
static constexpr int64_t MILLISECONDS_IN_DAY = SECONDS_IN_DAY * METRIC_FACTOR;
static constexpr int64_t MICROSECONDS_IN_DAY = MILLISECONDS_IN_DAY * METRIC_FACTOR;
static constexpr int64_t NANOSECONDS_IN_DAY = MICROSECONDS_IN_DAY * METRIC_FACTOR;

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
        return cudf::unary::Launcher< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
    }
};


template <typename TypeFrom> // This can be either cudf::date32 or cudf::date64
struct CastDateTo_Dispatcher {
    // Cast to other date type
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_same<TypeTo, cudf::date32>::value ||
        std::is_same<TypeTo, cudf::date64>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        if( input->dtype == GDF_DATE64 && output->dtype == GDF_DATE32 ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, MILLISECONDS_IN_DAY> >::launch(input, output);
        else if( input->dtype == GDF_DATE32 && output->dtype == GDF_DATE64 ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, MILLISECONDS_IN_DAY> >::launch(input, output);
        else // both columns have same type
            return cudf::unary::Launcher< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
    }
    
    // Cast to cudf::timestamp
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_same<TypeTo, cudf::timestamp>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        if( input->dtype == GDF_DATE32 && ( output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_s ) ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, SECONDS_IN_DAY> >::launch(input, output);
        else if( input->dtype == GDF_DATE32 && ( output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_ms ) ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, MILLISECONDS_IN_DAY> >::launch(input, output);
        else if( input->dtype == GDF_DATE32 && ( output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_us ) ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, MICROSECONDS_IN_DAY> >::launch(input, output);
        else if( input->dtype == GDF_DATE32 && ( output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_ns ) ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, NANOSECONDS_IN_DAY> >::launch(input, output);
        else if( input->dtype == GDF_DATE64 && output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_us) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( input->dtype == GDF_DATE64 && output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_s) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( input->dtype == GDF_DATE64 && output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_ms) 
            return cudf::unary::Launcher< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
        else if( input->dtype == GDF_DATE64 && output->dtype == GDF_TIMESTAMP && output->dtype_info.time_unit == TIME_UNIT_ns) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR_SQ> >::launch(input, output);
        else
            return GDF_UNSUPPORTED_DTYPE;
    }

    // Cast to arithmetic, cudf::bool8, cudf::category, or cudf::nvstring_category
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_arithmetic<TypeTo>::value ||
        std::is_same<TypeTo, cudf::bool8>::value ||
        std::is_same<TypeTo, cudf::category>::value ||
        std::is_same<TypeTo, cudf::nvstring_category>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return cudf::unary::Launcher< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
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
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, SECONDS_IN_DAY> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ms ) && output->dtype == GDF_DATE32 ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, MILLISECONDS_IN_DAY> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_us ) && output->dtype == GDF_DATE32 ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, MICROSECONDS_IN_DAY> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ns ) && output->dtype == GDF_DATE32 ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, NANOSECONDS_IN_DAY> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_s ) && output->dtype == GDF_DATE64 ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ms ) && output->dtype == GDF_DATE64 ) 
            return cudf::unary::Launcher< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_us ) && output->dtype == GDF_DATE64 ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( ( input->dtype == GDF_TIMESTAMP && input->dtype_info.time_unit == TIME_UNIT_ns ) && output->dtype == GDF_DATE64 ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR_SQ> >::launch(input, output);
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
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && output->dtype_info.time_unit == TIME_UNIT_s ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_s && output->dtype_info.time_unit == TIME_UNIT_us ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR_SQ> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_us && output->dtype_info.time_unit == TIME_UNIT_s ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR_SQ> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_s && output->dtype_info.time_unit == TIME_UNIT_ns ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR_CUBE> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && output->dtype_info.time_unit == TIME_UNIT_s ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR_CUBE> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_us && output->dtype_info.time_unit == TIME_UNIT_ns ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && output->dtype_info.time_unit == TIME_UNIT_us ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && output->dtype_info.time_unit == TIME_UNIT_ns ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR_SQ> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ns && output->dtype_info.time_unit == TIME_UNIT_ms ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR_SQ> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_us && output->dtype_info.time_unit == TIME_UNIT_ms ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, DownCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else if( input->dtype_info.time_unit == TIME_UNIT_ms && output->dtype_info.time_unit == TIME_UNIT_us ) 
            return cudf::unary::Launcher<TypeFrom, TypeTo, UpCasting<TypeFrom, TypeTo, METRIC_FACTOR> >::launch(input, output);
        else
            return GDF_TIMESTAMP_RESOLUTION_MISMATCH;
    }

    // Cast to arithmetic, cudf::bool8, cudf::category, or cudf::nvstring_category
    template <typename TypeTo>
    typename std::enable_if_t<
        std::is_arithmetic<TypeTo>::value ||
        std::is_same<TypeTo, cudf::bool8>::value ||
        std::is_same<TypeTo, cudf::category>::value ||
        std::is_same<TypeTo, cudf::nvstring_category>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return cudf::unary::Launcher< TypeFrom, TypeTo, DeviceCast<TypeFrom, TypeTo> >::launch(input, output);
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
    
    // Cast from arithmetic, cudf::bool8, cudf::category, or cudf::nvstring_category
    template <typename TypeFrom>
    typename std::enable_if_t<
        std::is_arithmetic<TypeFrom>::value ||
        std::is_same<TypeFrom, cudf::bool8>::value ||
        std::is_same<TypeFrom, cudf::category>::value ||
        std::is_same<TypeFrom, cudf::nvstring_category>::value,
    gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return cudf::type_dispatcher(output->dtype,
                                    CastTo_Dispatcher<TypeFrom>{},
                                    input, output);
    }
};


gdf_error gdf_cast(gdf_column *input, gdf_column *output) {
    cudf::unary::handleChecksAndValidity(input, output);

    return cudf::type_dispatcher(input->dtype,
                                CastFrom_Dispatcher{},
                                input, output);
}
