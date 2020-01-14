/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <iostream>
#include <functional>

#include "cudf/fixed_point/fixed_point.hpp"

namespace cudf {
namespace fp {

namespace detail {
    // this function is for binary operations like + and - which when the exponent (scale)
    // differ for lhs and rhs, you take the max exponent of the two and shift the other 
    // fixed_point in order to have the same exponent
    template<typename Rep1, int Radix1,
             typename Rep2, int Radix2, typename Binop>
    fixed_point<Rep1, Radix1> max_exponent_binop(fixed_point<Rep1, Radix1>  const& lhs, 
                                                 fixed_point<Rep2, Radix2> const& rhs, 
                                                 Binop binop) {

        static_assert(std::is_same<Rep1, Rep2>::value, "Represenation types should be the same");
        static_assert(Radix1 == Radix2,                "Radix types should be the same");
        
        // if exponents (aka scales) are different
        if (lhs._scale > rhs._scale) {
            auto const rhs_shifted_value = detail::shift<Radix1>(rhs._value, lhs._scale - rhs._scale);
            return fixed_point<Rep1, Radix1>{scaled_integer<Rep1>(binop(lhs._value, rhs_shifted_value), lhs._scale)};
        } else if (rhs._scale > lhs._scale) {
            auto lhs_shifted_value = detail::shift<Radix1>(lhs._value, rhs._scale - lhs._scale);
            return fixed_point<Rep1, Radix1>{scaled_integer<Rep1>(binop(lhs_shifted_value, rhs._value), rhs._scale)};
        }

        // if exponents (aka scales) are the same
        return fixed_point<Rep1, Radix1>{scaled_integer<Rep1>(binop(lhs._value, rhs._value), lhs._scale)};
    }
}

// PLUS Operation
template<typename Rep1, int Radix1,
         typename Rep2, int Radix2>
fixed_point<Rep1, Radix1> operator+(fixed_point<Rep1, Radix1> const& lhs, 
                                    fixed_point<Rep2, Radix2> const& rhs) {
    return detail::max_exponent_binop(lhs, rhs, std::plus<>());
}

// MINUS Operation
template<typename Rep1, int Radix1,
         typename Rep2, int Radix2>
fixed_point<Rep1, Radix1> operator-(fixed_point<Rep1, Radix1> const& lhs, 
                                    fixed_point<Rep2, Radix2> const& rhs) {
    return detail::max_exponent_binop(lhs, rhs, std::minus<>());
}

// MULTIPLIES Operation
template<typename Rep1, int Radix1,
         typename Rep2, int Radix2>
fixed_point<Rep1, Radix1> operator*(fixed_point<Rep1, Radix1> const& lhs, 
                                    fixed_point<Rep2, Radix2> const& rhs) {

    static_assert(std::is_same<Rep1, Rep2>::value, "Represenation types should be the same");
    static_assert(Radix1 == Radix2,                "Radix types should be the same");
    
    return fixed_point<Rep1, Radix1>{scaled_integer<Rep1>(lhs._value * rhs._value, lhs._scale + rhs._scale)};
}

// DIVISION Operation
template<typename Rep1, int Radix1,
         typename Rep2, int Radix2>
fixed_point<Rep1, Radix1> operator/(fixed_point<Rep1, Radix1> const& lhs, 
                                    fixed_point<Rep2, Radix2> const& rhs) {

    static_assert(std::is_same<Rep1, Rep2>::value, "Represenation types should be the same");
    static_assert(Radix1 == Radix2,                "Radix types should be the same");
    
    return fixed_point<Rep1, Radix1>{scaled_integer<Rep1>(lhs._value / rhs._value, lhs._scale - rhs._scale)};
}

template <typename Rep, int Radix>
std::ostream& operator<<(std::ostream& os, fixed_point<Rep, Radix> const& si) {
    return os << si.get();
}

} // namespace fp
} // namespace cudf