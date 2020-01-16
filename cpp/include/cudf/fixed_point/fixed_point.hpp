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

#include <cmath>
#include <limits>
#include <functional>

namespace cudf {
namespace fp {

using scale_type = int32_t;

template <typename Rep, int Radix>
class fixed_point;

namespace detail {
    // perform this operation when constructing with - scale (after negating scale)
    template <int Radix, typename T>
    constexpr auto right_shift(T const& val, scale_type const& scale) {
        return val / std::pow(Radix, scale);
    }

    // perform this operation when constructing with + scale
    template <int Radix, typename T>
    constexpr auto left_shift(T const& val, scale_type const& scale) {
        return val * std::pow(Radix, scale);
    }

    // convenience generic shift function
    template <int Radix, typename T>
    constexpr auto shift(T const& val, scale_type const& scale) {
        return scale < 0 ? right_shift<Radix>(val, -scale)
                         : left_shift <Radix>(val,  scale);
    }

    // forward declare TODO might be unnecessary one file restructure is done
    template <typename Rep1, int Radix1, 
              typename Rep2, int Radix2, typename Binop>
    fixed_point<Rep1, Radix1> max_exponent_binop(fixed_point<Rep1, Radix1> const& lhs, 
                                                 fixed_point<Rep2, Radix2> const& rhs, 
                                                 Binop binop);
}

// helper struct for constructing fixed_point when value is already shifted
template <typename Rep>
struct scaled_integer{
    Rep value;
    scale_type scale;
    explicit scaled_integer(Rep v, scale_type s) : value(v), scale(s) {}
};


// Rep = representative type
template <typename Rep, int Radix>
class fixed_point {

    scale_type _scale;
    Rep        _value;

public:

    // CONSTRUCTORS
    template <typename T = Rep,
              typename std::enable_if_t<(std::numeric_limits<T>::is_integer
                                      || std::is_floating_point<T>::value)>* = nullptr>
    explicit fixed_point(T const& value, scale_type const& scale) :
        _value(detail::shift<Radix>(value, scale)),
        _scale(scale)
    {
    }

    explicit fixed_point(scaled_integer<Rep> s) :
        _value(s.value), 
        _scale(s.scale)
    {
    }

    // EXPLICIT CONVERSION OPERATOR
    template <typename U> // TODO SFINAE to make sure it is not a fixed_point type
    explicit constexpr operator U() const { 
        return detail::shift<Radix>(static_cast<U>(_value), -_scale);
    }

    auto get() const noexcept {
        int  const rounded_val          = _value / std::pow(Radix, _scale);
        bool const needs_floating_point = rounded_val * Radix * _scale != _value;
        return needs_floating_point ? static_cast<double>(*this) : static_cast<Rep>(*this);
    }

    template <typename Rep2, int Radix2>
    fixed_point<Rep2, Radix2>& operator+=(fixed_point<Rep2, Radix2> const& rhs) {
        *this = *this + rhs;
        return *this;
    }

    template <typename Rep2, int Radix2>
    fixed_point<Rep2, Radix2>& operator*=(fixed_point<Rep2, Radix2> const& rhs) {
        *this = *this * rhs; 
        return *this;
    }

    template <typename Rep2, int Radix2>
    fixed_point<Rep2, Radix2>& operator-=(fixed_point<Rep2, Radix2> const& rhs) {
        *this = *this - rhs; 
        return *this;
    }

    template <typename Rep2, int Radix2>
    fixed_point<Rep2, Radix2>& operator/=(fixed_point<Rep2, Radix2> const& rhs) {
        *this = *this / rhs; 
        return *this;
    }

    // enable access to _value & _scale
    template <typename Rep1, int Radix1, 
              typename Rep2, int Radix2, typename Binop>
    friend fixed_point<Rep1, Radix1> detail::max_exponent_binop(fixed_point<Rep1, Radix1> const& lhs, 
                                                                fixed_point<Rep2, Radix2> const& rhs, 
                                                                Binop binop);

    // enable access to _value & _scale
    template <typename Rep1, int Radix1, 
              typename Rep2, int Radix2>
    friend fixed_point<Rep1, Radix1> operator*(fixed_point<Rep1, Radix1> const& lhs, 
                                               fixed_point<Rep2, Radix2> const& rhs);

    // enable access to _value & _scale
    template <typename Rep1, int Radix1, 
              typename Rep2, int Radix2>
    friend fixed_point<Rep1, Radix1> operator/(fixed_point<Rep1, Radix1> const& lhs, 
                                               fixed_point<Rep2, Radix2> const& rhs);                                                  
};

} // namespace fp
} // namespace cudf
