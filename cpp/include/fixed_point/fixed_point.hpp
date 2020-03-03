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
#include <iostream>
#include <limits>

namespace numeric {

// This is a wrapper struct that enforces "strong typing"
// at the construction site of the type. No implicit
// conversions will be allowed and you will need to use the
// name of the type alias (i.e. scale_type{0})
template <typename T>
struct strong_typedef {
    T _t;
    explicit strong_typedef(T t) : _t(t) {}
    operator T() const { return _t; }
};

using scale_type = strong_typedef<int32_t>;

enum class Radix : int32_t {
    BASE_2  = 2,
    BASE_10 = 10
};

template <typename T>
constexpr inline auto is_supported_representation_type() {
  return std::is_same<T, int32_t>::value
      || std::is_same<T, int64_t>::value;
}

template <typename T>
constexpr inline auto is_supported_construction_value_type() {
  return std::is_integral      <T>::value
      || std::is_floating_point<T>::value;
}

namespace detail {

    // `exponent` comes from using scale_type = strong_typedef<int32_t>
    //  we still need Rep to know what type we want for accumulator of ipow
    template <typename Rep, Radix Base, typename T,
              typename std::enable_if_t<(std::is_same<int32_t, T>::value
                                      && is_supported_representation_type<Rep>())>* = nullptr>
    Rep ipow(T exponent) {
        // Integer Exponentiation by Squaring:
        // https://simple.wikipedia.org/wiki/Exponentiation_by_squaring
        // Note: this is the iterative equivalent of the recursive definition (faster)
        // Quick-bench: http://quick-bench.com/Wg7o7HYQC9FW5M0CO0wQAjSwP_Y
        if (exponent == 0) return static_cast<Rep>(Base);
        auto extra  = static_cast<Rep>(1);
        auto square = static_cast<Rep>(Base);
        while (exponent > 1) {
            if (exponent & 1 /* odd */) {
                extra *= square;
                exponent -= 1;
            }
            exponent /= 2;
            square *= square;
        }
        return square * extra;
    }

    // helper function to negate strongly typed scale_type
    CUDA_HOST_DEVICE_CALLABLE
    auto negate(scale_type const& scale) {
        return scale_type{-scale};
    }

    // perform this operation when constructing with positive scale
    template <typename Rep, Radix Rad, typename T>
    CUDA_HOST_DEVICE_CALLABLE
    constexpr T right_shift(T const& val, scale_type const& scale) {
        assert(scale > 0);
        return val / ipow<Rep, Rad>(scale._t);
    }

    // perform this operation when constructing with negative scale
    template <typename Rep, Radix Rad, typename T>
    CUDA_HOST_DEVICE_CALLABLE
    constexpr T left_shift(T const& val, scale_type const& scale) {
        assert(scale < 0);
        return val * ipow<Rep, Rad>(-scale._t);
    }

    // convenience generic shift function
    template <typename Rep, Radix Rad, typename T>
    CUDA_HOST_DEVICE_CALLABLE
    constexpr T shift(T const& val, scale_type const& scale) {
        if      (scale == 0) return val;
        else if (scale >  0) return right_shift<Rep, Rad>(val, scale);
        else                 return left_shift <Rep, Rad>(val, scale);
    }

    template <typename Rep, Radix Rad, typename T,
              typename std::enable_if_t<is_supported_construction_value_type<T>()>* = nullptr>
    auto shift_with_precise_round(T const& value, scale_type const& scale) -> int64_t {
        if (scale == 0) return value;
        int64_t const base   = static_cast<int64_t>(Rad);
        int64_t const factor = ipow<int64_t, Rad>(std::abs(scale));
        int64_t const temp   = scale <= 0 ? value * (factor * base) : value / (factor / base);
        return std::roundf(static_cast<double>(temp) / base);
    }

}

/**
 * @brief Helper struct for constructing `fixed_point` when value is already
 * shifted
 *
 * example: using decimal32 = fixed_point<int32_t, Radix::BASE_10>;
 *          auto n = decimal32{scaled_integer{1001, 3}}; // n = 1.001
 *
 * @tparam Rep The representation type (either `int32_t` or `int64_t`)
 */
template <typename Rep,
          typename std::enable_if_t<is_supported_representation_type<Rep>()>* = nullptr>
struct scaled_integer{
    Rep value;
    scale_type scale;
    explicit scaled_integer(Rep v, scale_type s) : value(v), scale(s) {}
};

/**
 * @brief A type for representing a number with a fixed amount of precision
 *
 * Currently, only binary and decimal `fixed_point` numbers are supported.
 * Binary operations can only be performed with other `fixed_point` numbers
 *
 * @tparam Rep The representation type (either `int32_t` or `int64_t`)
 * @tparam Rad The radix/base (either `Radix::BASE_2` or `Radix::BASE_10`)
 */
template <typename Rep, Radix Rad>
class fixed_point {

    Rep        _value;
    scale_type _scale;

public:

    /**
    * @brief Constructor that will perform shifting to store value appropriately
    *
    * @tparam T The type that you are constructing from (integral or floating)
    * @param value The value that will be constructed from
    * @param scale The exponent that is applied to Rad to perform shifting
    */
    template <typename T,
              typename std::enable_if_t<is_supported_construction_value_type<T>()
                                     && is_supported_representation_type<Rep>()>* = nullptr>
    explicit fixed_point(T const& value, scale_type const& scale) :
        _value{static_cast<Rep>(detail::shift_with_precise_round<Rep, Rad>(value, scale))},
        _scale{scale}
    {
    }

    /**
    * @brief Constructor that will not perform shifting (assumes value already
    * shifted)
    *
    * @param s scaled_integer that contains scale and already shifted value
    */
    explicit fixed_point(scaled_integer<Rep> s) :
        _value{s.value},
        _scale{s.scale}
    {
    }

   /**
    * @brief Default constructor that constructs `fixed_point` number with a
    * value and scale of zero
    */
    fixed_point() :
        _value{0},
        _scale{scale_type{0}}
    {
    }

   /**
    * @brief Explicit conversion operator
    *
    * @tparam U The type that is being explicitly converted to (integral or floating)
    * @return The `fixed_point` number in base 10 (aka human readable format)
    */
    template <typename U,
              typename std::enable_if_t<is_supported_construction_value_type<U>()>* = nullptr>
    CUDA_HOST_DEVICE_CALLABLE
    explicit constexpr operator U() const {
        return detail::shift<Rep, Rad>(static_cast<U>(_value), detail::negate(_scale));
    }

    /**
    * @brief operator +=
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return The sum
    */
    template <typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    fixed_point<Rep1, Rad1>& operator+=(fixed_point<Rep1, Rad1> const& rhs) {
        *this = *this + rhs;
        return *this;
    }

    /**
    * @brief operator *=
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return The product
    */
    template <typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    fixed_point<Rep1, Rad1>& operator*=(fixed_point<Rep1, Rad1> const& rhs) {
        *this = *this * rhs;
        return *this;
    }

    /**
    * @brief operator -=
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return The difference
    */
    template <typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    fixed_point<Rep1, Rad1>& operator-=(fixed_point<Rep1, Rad1> const& rhs) {
        *this = *this - rhs;
        return *this;
    }

    /**
    * @brief operator /=
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return The quotient
    */
    template <typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    fixed_point<Rep1, Rad1>& operator/=(fixed_point<Rep1, Rad1> const& rhs) {
        *this = *this / rhs;
        return *this;
    }

    /**
    * @brief operator ++ (post-increment)
    *
    * @return The incremented result
    */
    CUDA_HOST_DEVICE_CALLABLE
    fixed_point<Rep, Rad>& operator++() {
        *this = *this + fixed_point<Rep, Rad>{1, scale_type{_scale}};
        return *this;
    }

    /**
    * @brief operator + (for adding two `fixed_point` numbers)
    *
    * If `_scale`s are equal, `_value`s are added
    * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
    * greater `_scale`, and then `_value`s are added
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return The resulting `fixed_point` sum
    */
    template <typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    friend fixed_point<Rep1, Rad1> operator+(fixed_point<Rep1, Rad1> const& lhs,
                                             fixed_point<Rep1, Rad1> const& rhs);

    /**
    * @brief operator - (for subtracting two `fixed_point` numbers)
    *
    * If `_scale`s are equal, `_value`s are substracted
    * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
    * greater `_scale`, and then `_value`s are substracted
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return The resulting `fixed_point` difference
    */
    template <typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    friend fixed_point<Rep1, Rad1> operator-(fixed_point<Rep1, Rad1> const& lhs,
                                             fixed_point<Rep1, Rad1> const& rhs);

    /**
    * @brief operator * (for multiplying two `fixed_point` numbers)
    *
    * `_scale`s are added and `_value`s are multiplied
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return The resulting `fixed_point` product
    */
    template <typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    friend fixed_point<Rep1, Rad1> operator*(fixed_point<Rep1, Rad1> const& lhs,
                                             fixed_point<Rep1, Rad1> const& rhs);

    /**
    * @brief operator / (for dividing two `fixed_point` numbers)
    *
    * `_scale`s are substracted and `_value`s are divided
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return The resulting `fixed_point` quotient
    */
    template <typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    friend fixed_point<Rep1, Rad1> operator/(fixed_point<Rep1, Rad1> const& lhs,
                                             fixed_point<Rep1, Rad1> const& rhs);

    /**
    * @brief operator == (for comparing two `fixed_point` numbers)
    *
    * If `_scale`s are equal, `_value`s are compared
    * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
    * greater `_scale`, and then `_value`s are compared
    *
    * @tparam Rep1 Representation type of number being added to `this`
    * @tparam Rad1 Radix (base) type of number being added to `this`
    * @return true if `lhs` and `rhs` are equal, false if not
    */
    template<typename Rep1, Radix Rad1>
    CUDA_HOST_DEVICE_CALLABLE
    friend bool operator==(fixed_point<Rep1, Rad1> const& lhs,
                           fixed_point<Rep1, Rad1> const& rhs);
};

template <typename Rep>
std::string print_rep() {
    if      (std::is_same<Rep, int8_t >::value) return "int8_t";
    else if (std::is_same<Rep, int16_t>::value) return "int16_t";
    else if (std::is_same<Rep, int32_t>::value) return "int32_t";
    else if (std::is_same<Rep, int64_t>::value) return "int64_t";
    else                                        return "unknown type";
}

template <typename Rep, typename T>
CUDA_HOST_DEVICE_CALLABLE
auto addition_overflow(T lhs, T rhs) {
    return rhs > 0 ? lhs > std::numeric_limits<Rep>::max() - rhs
                   : lhs < std::numeric_limits<Rep>::min() - rhs;
}

template <typename Rep, typename T>
CUDA_HOST_DEVICE_CALLABLE
auto subtraction_overflow(T lhs, T rhs) {
    return rhs > 0 ? lhs < std::numeric_limits<Rep>::min() + rhs
                   : lhs > std::numeric_limits<Rep>::max() + rhs;
}

template <typename Rep, typename T>
CUDA_HOST_DEVICE_CALLABLE
auto division_overflow(T lhs, T rhs) {
    return lhs == std::numeric_limits<Rep>::min() && rhs == -1;
}

template <typename Rep, typename T>
CUDA_HOST_DEVICE_CALLABLE
auto multiplication_overflow(T lhs, T rhs) {
    auto const min = std::numeric_limits<Rep>::min();
    auto const max = std::numeric_limits<Rep>::max();
    if      (rhs >  0) return lhs > max / rhs || lhs < min / rhs;
    else if (rhs < -1) return lhs > min / rhs || lhs < max / rhs;
    else               return rhs == -1 && lhs == min;
}

// PLUS Operation
template<typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE
fixed_point<Rep1, Rad1> operator+(fixed_point<Rep1, Rad1> const& lhs,
                                  fixed_point<Rep1, Rad1> const& rhs) {

    auto const rhsv  = lhs._scale > rhs._scale ? detail::shift_with_precise_round<Rep1, Rad1>(rhs._value, scale_type{lhs._scale - rhs._scale}) : rhs._value;
    auto const lhsv  = lhs._scale < rhs._scale ? detail::shift_with_precise_round<Rep1, Rad1>(lhs._value, scale_type{rhs._scale - lhs._scale}) : lhs._value;
    auto const scale = lhs._scale > rhs._scale ? lhs._scale : rhs._scale;

    #if defined(__CUDACC_DEBUG__)

    assert(!addition_overflow<Rep1>(lhsv, rhsv) &&
        "fixed_point overflow of underlying represenation type " + print_rep<Rep1>());

    #endif

    return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>(lhsv + rhsv, scale)};
}

// MINUS Operation
template<typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE
fixed_point<Rep1, Rad1> operator-(fixed_point<Rep1, Rad1> const& lhs,
                                  fixed_point<Rep1, Rad1> const& rhs) {

    auto const rhsv  = lhs._scale > rhs._scale ? detail::shift_with_precise_round<Rep1, Rad1>(rhs._value, scale_type{lhs._scale - rhs._scale}) : rhs._value;
    auto const lhsv  = lhs._scale < rhs._scale ? detail::shift_with_precise_round<Rep1, Rad1>(lhs._value, scale_type{rhs._scale - lhs._scale}) : lhs._value;
    auto const scale = lhs._scale > rhs._scale ? lhs._scale : rhs._scale;

    #if defined(__CUDACC_DEBUG__)

    assert(!subtraction_overflow<Rep1>(lhsv, rhsv) &&
        "fixed_point overflow of underlying represenation type " + print_rep<Rep1>());

    #endif

    return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>(lhsv - rhsv, scale)};
}

// MULTIPLIES Operation
template<typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE
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
CUDA_HOST_DEVICE_CALLABLE
fixed_point<Rep1, Rad1> operator/(fixed_point<Rep1, Rad1> const& lhs,
                                  fixed_point<Rep1, Rad1> const& rhs) {

    #if defined(__CUDACC_DEBUG__)

    assert(!division_overflow<Rep1>(lhs._value, rhs._value) &&
        "fixed_point overflow of underlying represenation type " + print_rep<Rep1>());

    #endif

    return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>(std::roundf(lhs._value * 1.0 / rhs._value), scale_type{lhs._scale - rhs._scale})};
}

// EQUALITY COMPARISON Operation
template<typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE
bool operator==(fixed_point<Rep1, Rad1> const& lhs,
                fixed_point<Rep1, Rad1> const& rhs) {
    auto const rhsv = lhs._scale > rhs._scale ? detail::shift_with_precise_round<Rep1, Rad1>(rhs._value, scale_type{lhs._scale - rhs._scale}) : rhs._value;
    auto const lhsv = lhs._scale < rhs._scale ? detail::shift_with_precise_round<Rep1, Rad1>(lhs._value, scale_type{rhs._scale - lhs._scale}) : lhs._value;
    return rhsv == lhsv;
}

/**
* @brief ostream operator for outputting `fixed_point` number
*
* @tparam Rep Representation type of number being output
* @tparam Rad Radix (base) type of number being output
* @param os target ostream
* @return fp `fixed_point` number being output
*/
template <typename Rep, Radix Radix>
std::ostream& operator<<(std::ostream& os, fixed_point<Rep, Radix> const& fp) {
    return os << static_cast<double>(fp);
}

} // namespace numeric
