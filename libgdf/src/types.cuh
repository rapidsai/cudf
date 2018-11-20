#ifndef GDF_CPPTYPES_H
#define GDF_CPPTYPES_H

#include <gdf/cffi/types.h>

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Placeholder type definitions for the non-primitive types for use in
 * conjunction with the gdf_type_dispatcher.
 *
 * The purpose of these type definitions is to allow distinguishing columns with
 * different gdf_dtype types, but have the same underlying type. For example,
 * the underlying type of both GDF_DATE32 and GDF_INT32 is int32_t. However, if
 * one wished to specialize a functor invoked with the gdf_type_dispatcher to handle
 * GDF_DATE32 different from GDF_INT32, that would not be possible without these
 * type definitions.
 *
 * In the long term, these types should be updated such that they provide appropriate
 * operators/member functions that implement the desired behavior for operating on
 * elements of these types.
 *
 */
/* ----------------------------------------------------------------------------*/
namespace gdf
{
  struct category
  {
    using value_type = gdf_category;
    value_type value;
  };

  struct timestamp
  {
    using value_type = gdf_timestamp;
    value_type value;
  };

  struct date32
  {
    using value_type = gdf_date32;
    value_type value;
  };

  struct date64
  {
    using value_type = gdf_date64;
    value_type value;
  };

  // TODO Add a type for GDF_STRING?


  namespace detail
  {

    template <typename T>
    struct needs_unwrap{ static constexpr bool value{false}; };
    template <> struct needs_unwrap<category>{static constexpr bool value{true}; };
    template <> struct needs_unwrap<timestamp>{static constexpr bool value{true}; };
    template <> struct needs_unwrap<date32>{static constexpr bool value{true}; };
    template <> struct needs_unwrap<date64>{static constexpr bool value{true}; };

  
    template <typename T>
    __host__ __device__ __forceinline__
    typename std::enable_if< needs_unwrap< typename std::decay<T>::type >::value, 
                             typename T::value_type>::type& 
    unwrap(T&& wrapped)
    {
      return wrapped.value;
    }

    template <typename T>
    __host__ __device__ __forceinline__
    typename std::enable_if< std::is_fundamental<typename std::decay<T>::type>::value, 
                             typename T::value_type>::type& 
    unwrap(T&& value)
    {
      return value;
    }
  } // namespace detail
} // namespace gdf

#endif
