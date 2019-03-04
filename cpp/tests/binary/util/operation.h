/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#ifndef GDF_TESTS_BINARY_OPERATION_UTIL_OPERATION_H
#define GDF_TESTS_BINARY_OPERATION_UTIL_OPERATION_H

#include <cmath>
#include <type_traits>
#include <cstdint>

namespace cudf {
namespace library {
namespace operation {

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Add {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            using TypeCommon = typename std::common_type<TypeLhs, TypeRhs>::type;
            return (TypeOut)((TypeCommon)lhs + (TypeCommon)rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Sub {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            using TypeCommon = typename std::common_type<TypeLhs, TypeRhs>::type;
            return (TypeOut)((TypeCommon)lhs - (TypeCommon)rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Mul {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            using TypeCommon = typename std::common_type<TypeLhs, TypeRhs>::type;
            return (TypeOut)((TypeCommon)lhs * (TypeCommon)rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Div {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            using TypeCommon = typename std::common_type<TypeLhs, TypeRhs>::type;
            return (TypeOut)((TypeCommon)lhs / (TypeCommon)rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct TrueDiv {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return (TypeOut)((double)lhs / (double)rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct FloorDiv {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return (TypeOut)floor((double)lhs / (double)rhs);
        }
    };

    template <typename TypeOut,
              typename TypeLhs,
              typename TypeRhs,
              typename Common = typename std::common_type<TypeLhs, TypeRhs>::type>
    struct Mod;

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Mod<TypeOut, TypeLhs, TypeRhs, int64_t> {
        TypeOut operator()(TypeLhs x, TypeRhs y) {
            return (TypeOut)((int64_t)x % (int64_t)y);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Mod<TypeOut, TypeLhs, TypeRhs, float> {
        TypeOut operator()(TypeLhs x, TypeRhs y) {
            return (TypeOut)fmod((float)x, (float)y);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Mod<TypeOut, TypeLhs, TypeRhs, double> {
        TypeOut operator()(TypeLhs x, TypeRhs y) {
            return (TypeOut)fmod((double)x, (double)y);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Pow {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return (TypeOut)pow((double)lhs, (double)rhs);
        }
    };

}  // namespace operation
}  // namespace library
}  // namespace cudf

#endif
