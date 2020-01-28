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
#include <cassert>
#include <functional>

namespace cudf {
namespace fixed_point {

template <typename T>
struct weak_typedef {
    T _t;
    explicit weak_typedef(T t) : _t(t) {}
    operator T() const { return _t; }
};

using scale_type = weak_typedef<int32_t>;

enum Radix : int32_t {
    BASE_2  = 2,
    BASE_10 = 10
};

namespace detail {
    // helper function to negate strongly typed scale_type
    auto negate(scale_type const& scale) -> scale_type {
        return scale_type{-scale};
    }

    // perform this operation when constructing with positive scale
    template <Radix Rad, typename T>
    constexpr auto right_shift(T const& val, scale_type const& scale) -> double {
        assert(scale > 0);
        return val / std::pow(static_cast<int32_t>(Rad), static_cast<int32_t>(scale));
    }

    // perform this operation when constructing with negative scale
    template <Radix Rad, typename T>
    constexpr auto left_shift(T const& val, scale_type const& scale) -> double {
        assert(scale < 0);
        return val * std::pow(static_cast<int32_t>(Rad), static_cast<int32_t>(negate(scale)));
    }

    // convenience generic shift function
    template <Radix Rad, typename T>
    constexpr auto shift(T const& val, scale_type const& scale) -> double {
        if      (scale == 0) return static_cast<double>(val);
        else if (scale >  0) return right_shift<Rad>(val, scale);
        else                 return left_shift <Rad>(val, scale);
    }
}

// helper struct for constructing fixed_point when value is already shifted
template <typename Rep>
struct scaled_integer{
    Rep value;
    scale_type scale;
    explicit scaled_integer(Rep v, scale_type s) : value(v), scale(s) {}
};


// Rep = representative type
template <typename Rep, Radix Rad>
class fixed_point {

    scale_type _scale;
    Rep        _value;

public:

    // CONSTRUCTORS
    template <typename T = Rep,
              typename std::enable_if_t<(std::is_integral      <T>::value
                                      || std::is_floating_point<T>::value)>* = nullptr>
    explicit fixed_point(T const& value, scale_type const& scale) :
        _value(detail::shift<Rad>(value, scale)),
        _scale(scale)
    {
    }

    explicit fixed_point(scaled_integer<Rep> s) :
        _value(s.value),
        _scale(s.scale)
    {
    }

    // DEFAULT CONSTRUCTOR
    fixed_point() :
        _value(0),
        _scale(scale_type{0})
    {
    }

    // EXPLICIT CONVERSION OPERATOR
    template <typename U,
              typename std::enable_if_t<(std::numeric_limits<U>::is_integer
                                      || std::is_floating_point<U>::value)>* = nullptr>
    explicit constexpr operator U() const {
        return detail::shift<Rad>(static_cast<U>(_value), detail::negate(_scale));
    }

    auto get() const noexcept {
        return static_cast<double>(*this);
    }

    template <typename Rep1, Radix Rad1>
    fixed_point<Rep1, Rad1>& operator+=(fixed_point<Rep1, Rad1> const& rhs) {
        *this = *this + rhs;
        return *this;
    }

    template <typename Rep1, Radix Rad1>
    fixed_point<Rep1, Rad1>& operator*=(fixed_point<Rep1, Rad1> const& rhs) {
        *this = *this * rhs;
        return *this;
    }

    template <typename Rep1, Radix Rad1>
    fixed_point<Rep1, Rad1>& operator-=(fixed_point<Rep1, Rad1> const& rhs) {
        *this = *this - rhs;
        return *this;
    }

    template <typename Rep1, Radix Rad1>
    fixed_point<Rep1, Rad1>& operator/=(fixed_point<Rep1, Rad1> const& rhs) {
        *this = *this / rhs;
        return *this;
    }

    fixed_point<Rep, Rad>& operator++() {
        *this = *this + fixed_point<Rep, Rad>{1, scale_type{_scale}};
        return *this;
    }

    // enable access to _value & _scale
    template <typename Rep1, Radix Rad1>
    friend fixed_point<Rep1, Rad1> operator+(fixed_point<Rep1, Rad1> const& lhs,
                                             fixed_point<Rep1, Rad1> const& rhs);

    // enable access to _value & _scale
    template <typename Rep1, Radix Rad1>
    friend fixed_point<Rep1, Rad1> operator-(fixed_point<Rep1, Rad1> const& lhs,
                                             fixed_point<Rep1, Rad1> const& rhs);

    // enable access to _value & _scale
    template <typename Rep1, Radix Rad1>
    friend fixed_point<Rep1, Rad1> operator*(fixed_point<Rep1, Rad1> const& lhs,
                                             fixed_point<Rep1, Rad1> const& rhs);

    // enable access to _value & _scale
    template <typename Rep1, Radix Rad1>
    friend fixed_point<Rep1, Rad1> operator/(fixed_point<Rep1, Rad1> const& lhs,
                                             fixed_point<Rep1, Rad1> const& rhs);
};

template <typename Rep>
auto print_rep() -> std::string {
    if      (std::is_same<Rep, int8_t >::value) return "int8_t";
    else if (std::is_same<Rep, int16_t>::value) return "int16_t";
    else if (std::is_same<Rep, int32_t>::value) return "int32_t";
    else if (std::is_same<Rep, int64_t>::value) return "int64_t";
    else                                        return "unknown type";
}

template <typename Rep, typename T>
auto addition_overflow(T lhs, T rhs) -> bool {
    return rhs > 0 ? lhs > std::numeric_limits<Rep>::max() - rhs
                   : lhs < std::numeric_limits<Rep>::min() - rhs;
}

template <typename Rep, typename T>
auto subtraction_overflow(T lhs, T rhs) -> bool {
    return rhs > 0 ? lhs < std::numeric_limits<Rep>::min() + rhs
                   : lhs > std::numeric_limits<Rep>::max() + rhs;
}

template <typename Rep, typename T>
auto division_overflow(T lhs, T rhs) -> bool {
    return lhs == std::numeric_limits<Rep>::min() && rhs == -1;
}

template <typename Rep, typename T>
auto multiplication_overflow(T lhs, T rhs) -> bool {
    auto const min = std::numeric_limits<Rep>::min();
    auto const max = std::numeric_limits<Rep>::max();
    if      (rhs >  0) return lhs > max / rhs || lhs < min / rhs;
    else if (rhs < -1) return lhs > min / rhs || lhs < max / rhs;
    else               return rhs == -1 && lhs == min;
}

// PLUS Operation
template<typename Rep1, Radix Rad1>
fixed_point<Rep1, Rad1> operator+(fixed_point<Rep1, Rad1> const& lhs,
                                  fixed_point<Rep1, Rad1> const& rhs) {

    auto const rhsv  = lhs._scale > rhs._scale ? detail::shift<Rad1>(rhs._value, scale_type{lhs._scale - rhs._scale}) : rhs._value;
    auto const lhsv  = lhs._scale < rhs._scale ? detail::shift<Rad1>(lhs._value, scale_type{rhs._scale - lhs._scale}) : lhs._value;
    auto const scale = lhs._scale > rhs._scale ? lhs._scale : rhs._scale;

    #if defined(__CUDACC_DEBUG__)

    assert(!addition_overflow<Rep1>(lhsv, rhsv) &&
        "fixed_point overflow of underlying represenation type " + print_rep<Rep1>());

    #endif

    return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>(lhsv + rhsv, scale)};
}

// MINUS Operation
template<typename Rep1, Radix Rad1>
fixed_point<Rep1, Rad1> operator-(fixed_point<Rep1, Rad1> const& lhs,
                                  fixed_point<Rep1, Rad1> const& rhs) {

    auto const rhsv  = lhs._scale > rhs._scale ? detail::shift<Rad1>(rhs._value, scale_type{lhs._scale - rhs._scale}) : rhs._value;
    auto const lhsv  = lhs._scale < rhs._scale ? detail::shift<Rad1>(lhs._value, scale_type{rhs._scale - lhs._scale}) : lhs._value;
    auto const scale = lhs._scale > rhs._scale ? lhs._scale : rhs._scale;

    #if defined(__CUDACC_DEBUG__)

    assert(!subtraction_overflow<Rep1>(lhsv, rhsv) &&
        "fixed_point overflow of underlying represenation type " + print_rep<Rep1>());

    #endif

    return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>(lhsv - rhsv, scale)};
}

// MULTIPLIES Operation
template<typename Rep1, Radix Rad1>
fixed_point<Rep1, Rad1> operator*(fixed_point<Rep1, Rad1> const& lhs,
                                  fixed_point<Rep1, Rad1> const& rhs) {

    #if defined(__CUDACC_DEBUG__)

    assert(!multiplication_overflow<Rep1>(lhs._value, rhs._value) &&
        "fixed_point overflow of underlying represenation type " + print_rep<Rep1>());

    #endif

    return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>(lhs._value * rhs._value, scale_type{lhs._scale + rhs._scale})};
}

// DIVISION Operation
template<typename Rep1, Radix Rad1>
fixed_point<Rep1, Rad1> operator/(fixed_point<Rep1, Rad1> const& lhs,
                                  fixed_point<Rep1, Rad1> const& rhs) {

    #if defined(__CUDACC_DEBUG__)

    assert(!division_overflow<Rep1>(lhs._value, rhs._value) &&
        "fixed_point overflow of underlying represenation type " + print_rep<Rep1>());

    #endif

    return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>(lhs._value / rhs._value, scale_type{lhs._scale - rhs._scale})};
}

// EQUALITY COMPARISON Operation
template<typename Rep1, Radix Rad1>
bool operator==(fixed_point<Rep1, Rad1> const& lhs,
                fixed_point<Rep1, Rad1> const& rhs) {
    auto const delta = std::fabs(lhs.get() - rhs.get());
    return delta < std::numeric_limits<decltype(delta)>::epsilon();
}

template <typename Rep, Radix Radix>
std::ostream& operator<<(std::ostream& os, fixed_point<Rep, Radix> const& si) {
    return os << si.get();
}

} // namespace fixed_point
} // namespace cudf
