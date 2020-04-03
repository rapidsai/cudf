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

#pragma once

#include <cmath>
#include <type_traits>
#include <cstdint>
#include <cudf/utilities/traits.hpp>

namespace cudf {
namespace library {
namespace operation {

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Add {
        // Disallow sum of timestamps with any other type (including itself)
        template <typename OutT = TypeOut,
                  typename std::enable_if<
                      !cudf::is_timestamp_t<OutT>::value &&
                      !cudf::is_timestamp_t<TypeLhs>::value &&
                      !cudf::is_timestamp_t<TypeRhs>::value, void>::type * = nullptr>
        OutT operator()(TypeLhs lhs, TypeRhs rhs) const {
            using TypeCommon = typename std::common_type<TypeLhs, TypeRhs>::type;
            return (static_cast<TypeCommon>(lhs) + static_cast<TypeCommon>(rhs));
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Sub {
        // Disallow difference of timestamps with any other type (including itself)
        template <typename OutT = TypeOut,
                  typename std::enable_if<
                      !cudf::is_timestamp_t<OutT>::value &&
                      !cudf::is_timestamp_t<TypeLhs>::value &&
                      !cudf::is_timestamp_t<TypeRhs>::value, void>::type * = nullptr>
        OutT operator()(TypeLhs lhs, TypeRhs rhs) const {
            return (static_cast<OutT>(lhs) - static_cast<OutT>(rhs));
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

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Equal {
        TypeOut operator()(TypeLhs x, TypeRhs y) {
            return (x == y);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Less {
        TypeOut operator()(TypeLhs x, TypeRhs y) {
            return (x < y);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct Greater {
        TypeOut operator()(TypeLhs x, TypeRhs y) {
            return (x > y);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct LessEqual {
        TypeOut operator()(TypeLhs x, TypeRhs y) {
            return (x <= y);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct GreaterEqual {
        TypeOut operator()(TypeLhs x, TypeRhs y) {
            return (x >= y);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct BitwiseAnd {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return (lhs & rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct BitwiseOr {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return (lhs | rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct BitwiseXor {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return (lhs ^ rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct LogicalAnd {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return TypeOut(lhs && rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct LogicalOr {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return TypeOut(lhs || rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct ShiftLeft {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return TypeOut(lhs << rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct ShiftRight {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return TypeOut(lhs >> rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct ShiftRightUnsigned {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return TypeOut(static_cast<std::make_unsigned_t<TypeLhs>>(lhs) >> rhs);
        }
    };

    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    struct LogBase {
        TypeOut operator()(TypeLhs lhs, TypeRhs rhs) {
            return TypeOut(std::log(static_cast<double>(lhs)) / std::log(static_cast<double>(rhs)));
        }
    };

}  // namespace operation
}  // namespace library
}  // namespace cudf
