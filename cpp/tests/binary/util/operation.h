/*
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

namespace gdf {
namespace library {
namespace operation {

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Add {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            using TypeCommon = typename std::common_type<TypeVax, TypeVay>::type;
            return (TypeOut)((TypeCommon)vax + (TypeCommon)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Sub {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            using TypeCommon = typename std::common_type<TypeVax, TypeVay>::type;
            return (TypeOut)((TypeCommon)vax - (TypeCommon)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mul {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            using TypeCommon = typename std::common_type<TypeVax, TypeVay>::type;
            return (TypeOut)((TypeCommon)vax * (TypeCommon)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Div {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            using TypeCommon = typename std::common_type<TypeVax, TypeVay>::type;
            return (TypeOut)((TypeCommon)vax / (TypeCommon)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct TrueDiv {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            return (TypeOut)((double)vax / (double)vay);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct FloorDiv {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            return (TypeOut)floor((double)vax / (double)vay);
        }
    };

    template <typename TypeOut,
              typename TypeVax,
              typename TypeVay,
              typename Common = typename std::common_type<TypeVax, TypeVay>::type>
    struct Mod;

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mod<TypeOut, TypeVax, TypeVay, uint64_t> {
        TypeOut operator()(TypeVax x, TypeVay y) {
            return (TypeOut)((uint64_t)x % (uint64_t)y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mod<TypeOut, TypeVax, TypeVay, float> {
        TypeOut operator()(TypeVax x, TypeVay y) {
            return (TypeOut)fmod((float)x, (float)y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Mod<TypeOut, TypeVax, TypeVay, double> {
        TypeOut operator()(TypeVax x, TypeVay y) {
            return (TypeOut)fmod((double)x, (double)y);
        }
    };

    template <typename TypeOut, typename TypeVax, typename TypeVay>
    struct Pow {
        TypeOut operator()(TypeVax vax, TypeVay vay) {
            return (TypeOut)pow((double)vax, (double)vay);
        }
    };

}  // namespace operation
}  // namespace library
}  // namespace gdf

#endif
