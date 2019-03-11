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

#include "unary_ops.cuh"

#include "utilities/type_dispatcher.hpp"

#include <cmath>
#include <algorithm>
#include <type_traits>

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

template <typename F>
struct MathOpDispatcher {
    template<typename T>
    static gdf_error launch(gdf_column *input, gdf_column *output) {
        return UnaryOp<T, T, F>::launch(input, output);
    }

    template <typename T>
    typename std::enable_if_t<std::is_floating_point<T>::value, gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return launch<T>(input, output);
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
