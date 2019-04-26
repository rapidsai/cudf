#ifndef GDF_CPPTYPES_H
#define GDF_CPPTYPES_H

#include <cudf/types.h>
#include "cudf_utils.h"

#include <cub/util_type.cuh>

#include <iosfwd>
#include <type_traits>
#include <limits>

/* --------------------------------------------------------------------------*/
/** 
 * @file wrapper_types.hpp
 * @brief  Wrapper structs for for the non-fundamental gdf_dtype types.
 *
 * These structs simply wrap a single member variable of a fundamental type 
 * called "value". 
 * 
 * These wrapper structures are used in conjunction with the type_dispatcher to
 * emulate "strong typedefs", i.e., provide opaque types that allow template 
 * specialization. A normal C++ typedef is simply an alias and does not allow
 * for specializing a template or overloading a function.
 * 
 * The purpose of these "strong typedefs" is to provide a one-to-one mapping between
 * gdf_dtype enum values and concrete C++ types and allow distinguishing columns with
 * different gdf_dtype types, but have the same underlying type. For example,
 * the underlying type of both GDF_DATE32 and GDF_INT32 is int32_t. However, if
 * one wished to specialize a functor invoked with the type_dispatcher to handle
 * GDF_DATE32 different from GDF_INT32, that would not be possible with aliases.
 * 
 * The standard arithmetic operators are defined for these wrapper structs such
 * that they can be used as if they were fundamental arithmetic types.
 * 
 * In general, interacting with the wrapper structs should be done via the defined 
 * operators. However, if one needs to directly access the underlying value, the
 * "unwrap" function may be used. Calling `unwrap` on an instance of a wrapper struct
 * will return a reference to the underlying value. Calling `unwrap` on an instance
 * of a fundamental type will return a reference to that instance (effectively a no-op).
 *
 */
/* ----------------------------------------------------------------------------*/
namespace cudf
{
namespace detail
{
/**
     * @brief Base wrapper structure to emulate "strong typedefs" for gdf_dtype values 
     * that do not correspond to fundamental types.
     * 
     * Implements operators that allow the wrapper to be used as if it were a fundamental
     * type.
     * 
     * @tparam T  The type of the wrapped value, i.e., the "underlying type" of the wrapper
     * @tparam type_id  The wrapped gdf_dtype
     */
template <typename T, gdf_dtype type_id>
struct wrapper
{
  static constexpr gdf_dtype corresponding_column_type{type_id}; ///< The wrapped gdf_dtype
  using value_type = T;                                          ///< The underlying fundamental type of the wrapper
  value_type value;                                              ///< The wrapped value

  CUDA_HOST_DEVICE_CALLABLE
  constexpr explicit wrapper(T v) : value{v} {}

  CUDA_HOST_DEVICE_CALLABLE
  explicit operator value_type() const { return this->value; }

  wrapper(wrapper const& w) = default;

  wrapper() = default;
};

template <typename T, gdf_dtype type_id>
std::ostream& operator<<(std::ostream& os, wrapper<T, type_id> const& w) 
{
  return os << w.value;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
bool operator==(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs) 
{ 
  return lhs.value == rhs.value; 
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
bool operator!=(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs) 
{ 
  return lhs.value != rhs.value; 
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
bool operator<=(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs) 
{ 
  return lhs.value <= rhs.value; 
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
bool operator>=(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs)
{ 
  return lhs.value >= rhs.value; 
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE 
bool operator<(wrapper<T, type_id> const &lhs, wrapper<T, type_id> const &rhs)
{
  return lhs.value < rhs.value;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE 
bool operator>(wrapper<T, type_id> const &lhs, wrapper<T, type_id> const &rhs)
{
  return lhs.value > rhs.value;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator+=(wrapper<T,type_id> & lhs, wrapper<T,type_id> const& rhs)
{
  lhs.value += rhs.value;
  return lhs;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator-=(wrapper<T,type_id> & lhs, wrapper<T,type_id> const& rhs)
{
  lhs.value -= rhs.value;
  return lhs;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator*=(wrapper<T,type_id> & lhs, wrapper<T,type_id> const& rhs)
{
  lhs.value *= rhs.value;
  return lhs;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator/=(wrapper<T,type_id> & lhs, wrapper<T,type_id> const& rhs)
{
  lhs.value /= rhs.value;
  return lhs;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator+(wrapper<T, type_id> const &lhs, wrapper<T, type_id> const &rhs)
{
  return wrapper<T, type_id>{lhs.value + rhs.value};
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator-(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs)
{
  return wrapper<T, type_id>{lhs.value - rhs.value};
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator*(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs)
{
  return wrapper<T, type_id>{lhs.value * rhs.value};
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator/(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs)
{
  return wrapper<T, type_id>{lhs.value / rhs.value};
}

// prefix increment operator
template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator++(wrapper<T,type_id> & w)
{
  w.value++;
  return w;
}

// postfix increment operator
template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator++(wrapper<T,type_id> & w, int)
{
  return wrapper<T,type_id>{w.value++};
}

// prefix decrement operator
template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator--(wrapper<T,type_id> & w)
{
  w.value--;
  return w;
}

// postfix decrement operator
template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator--(wrapper<T,type_id> & w, int)
{
  return wrapper<T,type_id>{w.value--};
}



/* --------------------------------------------------------------------------*/
/** 
     * @brief  Returns a reference to the underlying "value" member of a wrapper struct
     * 
     * @param[in] wrapped A non-const reference to the wrapper struct to unwrap
     * 
     * @returns A reference to the underlying wrapped value  
     */
/* ----------------------------------------------------------------------------*/
template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
    typename wrapper<T, type_id>::value_type &
    unwrap(wrapper<T, type_id> &wrapped)
{
  return wrapped.value;
}

/* --------------------------------------------------------------------------*/
/** 
     * @brief  Returns a reference to the underlying "value" member of a wrapper struct
     * 
     * @param[in] wrapped A const reference to the wrapper struct to unwrap
     * 
     * @returns A const reference to the underlying wrapped value  
     */
/* ----------------------------------------------------------------------------*/
template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
    typename wrapper<T, type_id>::value_type const &
    unwrap(wrapper<T, type_id> const &wrapped)
{
  return wrapped.value;
}

/* --------------------------------------------------------------------------*/
/** 
     * @brief Passthrough function for fundamental types
     *
     * This specialization of "unwrap" is provided such that it can be used in generic
     * code that is agnostic to whether or not the type being operated on is a wrapper
     * struct or a fundamental type
     * 
     * @param[in] value Reference to a fundamental type to passthrough
     * 
     * @returns Reference to the value passed in
     */
/* ----------------------------------------------------------------------------*/
template <typename T>
CUDA_HOST_DEVICE_CALLABLE
    typename std::enable_if_t<std::is_fundamental<typename std::decay<T>::type>::value,
                              T> &
    unwrap(T &value)
{
  return value;
}

/* --------------------------------------------------------------------------*/
/** 
     * @brief Passthrough function for fundamental types
     *
     * This specialization of "unwrap" is provided such that it can be used in generic
     * code that is agnostic to whether or not the type being operated on is a wrapper
     * struct or a fundamental type
     * 
     * @param[in] value const reference to a fundamental type to passthrough
     * 
     * @returns const reference to the value passed in
     */
/* ----------------------------------------------------------------------------*/
template <typename T>
CUDA_HOST_DEVICE_CALLABLE
    typename std::enable_if_t<std::is_fundamental<typename std::decay<T>::type>::value,
                              T> const &
    unwrap(T const &value)
{
  return value;
}

/**---------------------------------------------------------------------------*
 * @brief Trait to use to get underlying type of wrapped object
 * 
 * This struct can be used with either a fundamental type or a wrapper type and
 * it uses unwrap to get the underlying type.
 * 
 * Example use case: 
 *  Making a functor to use with a `type_dispatcher` that works on the
 *  underlying type of all `gdf_dtype`
 *  
 * ```c++
 * struct example_functor{
 *  template <typename T>
 *  int operator()(){
 *    using T1 = cudf::detail::unwrapped_type<T>::type;
 *    return sizeof(T1);
 *  }
 * };
 * ```
 * 
 * @tparam T Either wrapped object type or fundamental type
 *---------------------------------------------------------------------------**/
template <typename T>
struct unwrapped_type {
  using type = std::decay_t<decltype(unwrap(std::declval<T&>()))>;
};

/**---------------------------------------------------------------------------*
 * @brief Helper type for `unwrapped_type`
 * 
 * Example:
 * ```c++
 * using T1 = cudf::detail::unwrapped_type_t<date32>; // T1 = int 
 * using T2 = cudf::detail::unwrapped_type_t<float>;  // T2 = float 
 * ```
 * 
 * @tparam T Either wrapped object type or fundamental type
 *---------------------------------------------------------------------------**/
template <typename T>
using unwrapped_type_t = typename unwrapped_type<T>::type;

} // namespace detail

using category = detail::wrapper<gdf_category, GDF_CATEGORY>;

using nvstring_category = detail::wrapper<gdf_nvstring_category, GDF_STRING_CATEGORY>;

using timestamp = detail::wrapper<gdf_timestamp, GDF_TIMESTAMP>;

using date32 = detail::wrapper<gdf_date32, GDF_DATE32>;

using date64 = detail::wrapper<gdf_date64, GDF_DATE64>;

} // namespace cudf

namespace std
{

/**---------------------------------------------------------------------------*
 * @brief Specialization of std::numeric_limits for wrapper types
 *---------------------------------------------------------------------------**/
template <typename T, gdf_dtype type_id>
struct numeric_limits< cudf::detail::wrapper<T, type_id> > {
  
  using wrapper_t = cudf::detail::wrapper<T, type_id>;

  /**---------------------------------------------------------------------------*
   * @brief Returns the maximum finite value representable by the numeric type T
   *---------------------------------------------------------------------------**/
  static constexpr wrapper_t max() noexcept {
    return wrapper_t{ std::numeric_limits<T>::max() };
  }
  
  /**---------------------------------------------------------------------------*
   * @brief Returns the lowest finite value representable by the numeric type T
   * 
   * Returns a finite value x such that there is no other finite value y where y < x
   *---------------------------------------------------------------------------**/
  static constexpr wrapper_t lowest() noexcept {
    return wrapper_t{ std::numeric_limits<T>::lowest() };
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the minimum finite value representable by the numeric type T
   * 
   * For floating-point types with denormalization, min returns the minimum
   * positive normalized value.
   *---------------------------------------------------------------------------**/
  static constexpr wrapper_t min() noexcept {
    return wrapper_t{ std::numeric_limits<T>::min() };
  }

};

} // std

namespace cub
{

template <> struct NumericTraits<cudf::date32> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::date32>>,
    cudf::detail::unwrapped_type_t<cudf::date32>> {};

template <> struct NumericTraits<cudf::timestamp> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::timestamp>>,
    cudf::detail::unwrapped_type_t<cudf::timestamp>> {};

template <> struct NumericTraits<cudf::date64> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::date64>>,
    cudf::detail::unwrapped_type_t<cudf::date64>> {};

template <> struct NumericTraits<cudf::category> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::category>>,
    cudf::detail::unwrapped_type_t<cudf::category>> {};

template <> struct NumericTraits<cudf::nvstring_category> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::nvstring_category>>,
    cudf::detail::unwrapped_type_t<cudf::nvstring_category>> {};

} // cub

#endif
