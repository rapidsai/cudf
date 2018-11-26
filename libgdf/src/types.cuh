#ifndef GDF_CPPTYPES_H
#define GDF_CPPTYPES_H

#include <gdf/cffi/types.h>

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Wrapper structs for for the non-fundamental gdf_dtype types.
 *
 * These structs simply wrap a single member variable of a fundamental type 
 * called "value". In order to access the underlying wrapped value, one can 
 * use the "unwrap" function which will provide a reference to the underlying
 * value.
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
    static constexpr gdf_dtype element_type_id{GDF_CATEGORY};
    using value_type = gdf_category;
    value_type value;
  };

  struct timestamp
  {
    static constexpr gdf_dtype element_type_id{GDF_TIMESTAMP};
    using value_type = gdf_timestamp;
    value_type value;
  };

  struct date32
  {
    static constexpr gdf_dtype element_type_id{GDF_DATE32};
    using value_type = gdf_date32;
    value_type value;
  };

  struct date64
  {
    static constexpr gdf_dtype element_type_id{GDF_DATE64};
    using value_type = gdf_date64;
    value_type value;
  };

  namespace detail
  {
    /* --------------------------------------------------------------------------*/
    /** 
     * @brief  Returns a reference to the underlying "value" member of a wrapper struct
     * 
     * @Param[in] wrapped A non-const reference to the wrapper struct to unwrap
     * 
     * @Returns A reference to the underlying wrapped value  
     */
    /* ----------------------------------------------------------------------------*/
    template <typename T>
    __host__ __device__ __forceinline__
    typename std::enable_if_t< not std::is_fundamental<typename std::decay<T>::type>::value, 
                               typename std::decay<T>::type::value_type>& 
    unwrap(T& wrapped)
    {
      return wrapped.value;
    }

    /* --------------------------------------------------------------------------*/
    /** 
     * @brief  Returns a reference to the underlying "value" member of a wrapper struct
     * 
     * @Param[in] wrapped A const reference to the wrapper struct to unwrap
     * 
     * @Returns A const reference to the underlying wrapped value  
     */
    /* ----------------------------------------------------------------------------*/
    template <typename T>
    __host__ __device__ __forceinline__
    typename std::enable_if_t< not std::is_fundamental<typename std::decay<T>::type>::value, 
                               typename std::decay<T>::type::value_type> const& 
    unwrap(T const& wrapped)
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
     * @Param[in] value Reference to a fundamental type to passthrough
     * 
     * @Returns Reference to the value passed in
     */
    /* ----------------------------------------------------------------------------*/
    template <typename T>
    __host__ __device__ __forceinline__
    typename std::enable_if_t< std::is_fundamental< typename std::decay<T>::type >::value, 
                               T>& 
    unwrap(T& value)
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
     * @Param[in] value const reference to a fundamental type to passthrough
     * 
     * @Returns const reference to the value passed in
     */
    /* ----------------------------------------------------------------------------*/
    template <typename T>
    __host__ __device__ __forceinline__
    typename std::enable_if_t< std::is_fundamental< typename std::decay<T>::type >::value, 
                               T> const& 
    unwrap(T const& value)
    {
      return value;
    }
  } // namespace detail
} // namespace gdf

#endif
