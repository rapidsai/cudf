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

namespace cudf {
namespace binops {
namespace jit {
namespace code {

const char* operation =
R"***(
#pragma once
    #include "traits.h"

    struct Add {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<TypeOut>(x) + static_cast<TypeOut>(y));
        }
    };

    using RAdd = Add;

    struct Sub {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<TypeOut>(x) - static_cast<TypeOut>(y));
        }
    };

    struct RSub {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<TypeOut>(y) - static_cast<TypeOut>(x));
        }
    };

    struct Mul {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<TypeOut>(x) * static_cast<TypeOut>(y));
        }
    };

    using RMul = Mul;

    struct Div {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<TypeOut>(x) / static_cast<TypeOut>(y));
        }
    };

    struct RDiv {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<TypeOut>(y) / static_cast<TypeOut>(x));
        }
    };

    struct TrueDiv {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<double>(x) / static_cast<double>(y));
        }
    };

    struct RTrueDiv {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<double>(y) / static_cast<double>(x));
        }
    };

    struct FloorDiv {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return floor(static_cast<double>(x) / static_cast<double>(y));
        }
    };

    struct RFloorDiv {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return floor(static_cast<double>(y) / static_cast<double>(x));
        }
    };

    struct Mod {
        template <typename TypeOut,
                  typename TypeLhs,
                  typename TypeRhs,
                  enable_if_t<(is_integral_v<TypeOut>)>* = nullptr>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<TypeOut>(x) % static_cast<TypeOut>(y));
        }

        template <typename TypeOut,
                  typename TypeLhs,
                  typename TypeRhs,
                  enable_if_t<(isFloat<TypeOut>)>* = nullptr>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return fmodf(static_cast<TypeOut>(x), static_cast<TypeOut>(y));
        }

        template <typename TypeOut,
                  typename TypeLhs,
                  typename TypeRhs,
                  enable_if_t<(isDouble<TypeOut>)>* = nullptr>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return fmod(static_cast<TypeOut>(x), static_cast<TypeOut>(y));
        }
    };

    struct RMod {
        template <typename TypeOut,
                  typename TypeLhs,
                  typename TypeRhs,
                  enable_if_t<(is_integral_v<TypeOut>)>* = nullptr>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (static_cast<TypeOut>(y) % static_cast<TypeOut>(x));
        }

        template <typename TypeOut,
                  typename TypeLhs,
                  typename TypeRhs,
                  enable_if_t<(isFloat<TypeOut>)>* = nullptr>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return fmodf(static_cast<TypeOut>(y), static_cast<TypeOut>(x));
        }

        template <typename TypeOut,
                  typename TypeLhs,
                  typename TypeRhs,
                  enable_if_t<(isDouble<TypeOut>)>* = nullptr>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return fmod(static_cast<TypeOut>(y), static_cast<TypeOut>(x));
        }
    };

    struct Pow {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return pow(static_cast<double>(x), static_cast<double>(y));
        }
    };

    struct RPow {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return pow(static_cast<double>(y), static_cast<double>(x));
        }
    };

    struct Equal {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (x == y);
        }
    };

    using REqual = Equal;

    struct NotEqual {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (x != y);
        }
    };

    using RNotEqual = NotEqual;

    struct Less {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (x < y);
        }
    };

    struct RLess {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (y < x);
        }
    };

    struct Greater {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (x > y);
        }
    };

    struct RGreater {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (y > x);
        }
    };

    struct LessEqual {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (x <= y);
        }
    };

    struct RLessEqual {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (y <= x);
        }
    };

    struct GreaterEqual {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (x >= y);
        }
    };

    struct RGreaterEqual {
        template <typename TypeOut, typename TypeLhs, typename TypeRhs>
        static TypeOut operate(TypeLhs x, TypeRhs y) {
            return (y >= x);
        }
    };
)***";

} // namespace code
} // namespace jit
} // namespace binops
} // namespace cudf
