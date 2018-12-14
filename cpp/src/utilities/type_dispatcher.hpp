#ifndef GDF_TYPE_DISPATCHER_H
#define GDF_TYPE_DISPATCHER_H

#include <iostream>
#include "cudf/types.h"
#include "wrapper_types.hpp"
#include "NVStrings.h"
//#include "type_name.hpp"

#include <iostream>
#include <type_traits>
#include <cassert>
#include <utility>

namespace cudf {

namespace detail {

template <class>
struct sfinae_true : std::true_type{};

template <class F, typename T, typename... Ts>
static auto test_templated_invoke_operator(int) ->
    sfinae_true<decltype(std::declval<F>().template operator()<T, Ts...>(std::forward(std::declval<Ts>())... ))>;

template <class, typename, typename... Ts>
static auto test_templated_invoke_operator(long) -> std::false_type;

// A useful trait in itself:
template <class F, typename T, typename... Ts>
struct has_templated_invoke_operator : decltype( test_templated_invoke_operator<F,T, Ts...>(int{}) )
{ };

template <bool ActuallyInvoke, class F, typename T, typename... Ts>
struct invoke_if_possible_inner;

template <class F, typename T, typename... Ts>
struct invoke_if_possible_inner<false, F, T, Ts...>
{
    void operator()(std::integral_constant<bool, false>, F functor, Ts&&... params) { }
};

template <class F, typename T, typename... Ts>
struct invoke_if_possible_inner<true, F, T, Ts...>
{
    void operator()(std::integral_constant<bool, true>, F functor, Ts&&... params)
    {
        return functor.template operator()<T>(std::forward<Ts>(params)...);
    }
};

template <typename T>
struct invoke_if_possible {
    template <class F, typename... Ts>
    auto operator()(F functor, Ts&&... params)
    {
        constexpr auto actually_invoke = has_templated_invoke_operator<F, T, Ts...>::value;
        return detail::invoke_if_possible_inner<actually_invoke, F, T, Ts...>{}(functor, std::forward<Ts>(params)...);
    }
};

// As you read the above code (after the trait), you might be wondering:
//
// 1. Why is invoke_if_possible a struct, rather than a function? After all, the only thing we do
//    with it is invoke it (with operator())?
// 2. Why have two structs, an internal one (invoke_if_possible_inner) and an external one (invoke_if_possible)?
//    Isn't the internal one enough?
//
// Answers:
//
// 1. A C++ limitation. We could not have performed the tag dispatching with just functions, as
//    C++14 does not allow this kind of partial specialization for freestanding functions.
// 2. Note the template parameter on the outer struct. It allows client code to specify explicitly
//    which type it's interested in for instantiation of F's templated operator(), and
//    specify all other template parameters explicitly via the call to the outer struct's operator().
//    See the (only?) example of use in the type_dispatcher mechanism, below.
//
// Finally - note that much of this ugly TMP can easily go away if C++17 is available - using
// if constexpr to avoid SFINAE.


template<typename... Ts>
struct first_nonvoid_type;

template<typename T, typename... Us>
struct first_nonvoid_type<T, Us...> {
    using type = std::conditional_t<
        not std::is_same<typename std::decay<T>::type, void>::value,
        typename std::decay<T>::type,
        typename first_nonvoid_type<Us...>::type
    >;
};

template<>
struct first_nonvoid_type<> { using type = void; };

template<typename... Ts>
using first_nonvoid_type_t = typename first_nonvoid_type<Ts...>::type;

} // namespace detail


/**
 * @brief  Invokes an instance of a functor template with the appropriate type
 * determined by a gdf_dtype enum value.
 *
 * This helper function accepts any object with an "operator()" template,
 * e.g., a functor. It will invoke an instance of the template by passing
 * in as the template argument an appropriate type determined by the value of the
 * gdf_dtype argument.
 *
 * The template may have 1 or more template parameters, but the first parameter must
 * be the type dispatched from the gdf_dtype enum.  The remaining template parameters
 * must be able to be automatically deduced.
 *
 * There is a 1-to-1 mapping of gdf_dtype enum values and dispatched types. However,
 * different gdf_dtype values may have the same underlying type. Therefore, in
 * order to provide the 1-to-1 mapping, a wrapper struct may be dispatched for certain
 * gdf_dtype enum values in order to emulate a "strong typedef".
 *
 * A strong typedef  provides a new, concrete type unlike a normal C++ typedef which
 * is simply a type alias. These "strong typedef" structs simply wrap a single member
 * variable of a fundamental type called 'value'.
 *
 * The standard arithmetic operators are defined for the wrapper structs and therefore
 * the wrapper struct types can be used as if they were fundamental types.
 *
 * See wrapper_types.hpp for more detail.
 *
 * Example usage with a functor that returns the size of the dispatched type:
 *
 * struct example_functor{
 *  template <typename T>
 *  int operator()(){
 *    return sizeof(T);
 *  }
 * };
 *
 * cudf::type_dispatcher(GDF_INT8, example_functor);  // returns 1
 * cudf::type_dispatcher(GDF_INT64, example_functor); // returns 8
 *
 * Example usage of a functor for checking if element "i" in column "lhs" is
 * equal to element "j" in column "rhs":
 *
 * struct elements_are_equal{
 *   template <typename ColumnType>
 *   bool operator()(void const * lhs, int i,
 *                   void const * rhs, int j)
 *   {
 *     // Cast the void* data buffer to the dispatched type and retrieve elements
 *     // "i" and "j" from the respective columns
 *     ColumnType const i_elem = static_cast<ColumnType const*>(lhs)[i];
 *     ColumnType const j_elem = static_cast<ColumnType const*>(rhs)[j];
 *
 *     // operator== is defined for wrapper structs such that it performs the
 *     // operator== on the underlying values. Therefore, the wrapper structs
 *     // can be used as if they were fundamental arithmetic types
 *     return i_elem == j_elem;
 *   }
 * };
 *
 * The return type for all template instantiations of the functor's "operator()"
 * lambda must be the same, else there will be a compiler error as you would be
 * trying to return different types from the same function.
 *
 * @Param dtype The gdf_dtype enum that determines which type will be dispatched
 * @Param f The functor with a templated "operator()" that will be invoked with
 * the dispatched type
 * @Param args A parameter-pack (i.e., arbitrary number of arguments) that will
 * be perfectly-forwarded as the arguments of the functor's "operator()".
 *
 * @Returns Whatever is returned by the functor's "operator()".
 *
 */
// This pragma disables a compiler warning that complains about the valid usage	
// of calling a __host__ functor from this function which is __host__ __device__
#pragma hd_warning_disable
template < class functor_t, 
           typename... Ts>
CUDA_HOST_DEVICE_CALLABLE
constexpr auto type_dispatcher(gdf_dtype dtype,
                               functor_t f, 
                               Ts&&... args)
{
  using fallback_return_type = typename detail::first_nonvoid_type_t<
      decltype(f.template operator()< int8_t    >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< int16_t   >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< int32_t   >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< int64_t   >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< float     >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< double    >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< date32    >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< date64    >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< timestamp >(std::forward<Ts>(args)...)),
      decltype(f.template operator()< category  >(std::forward<Ts>(args)...))
  >;
  switch(dtype)
  {
    // The .template is known as a "template disambiguator" 
    // See here for more information: https://stackoverflow.com/questions/3786360/confusing-template-error
    case GDF_INT8:      { return f.template operator()< int8_t    >(std::forward<Ts>(args)...); }
    case GDF_INT16:     { return f.template operator()< int16_t   >(std::forward<Ts>(args)...); }
    case GDF_INT32:     { return f.template operator()< int32_t   >(std::forward<Ts>(args)...); }
    case GDF_INT64:     { return f.template operator()< int64_t   >(std::forward<Ts>(args)...); }
    case GDF_FLOAT32:   { return f.template operator()< float     >(std::forward<Ts>(args)...); }
    case GDF_FLOAT64:   { return f.template operator()< double    >(std::forward<Ts>(args)...); }
    case GDF_DATE32:    { return f.template operator()< date32    >(std::forward<Ts>(args)...); }
    case GDF_DATE64:    { return f.template operator()< date64    >(std::forward<Ts>(args)...); }
    case GDF_TIMESTAMP: { return f.template operator()< timestamp >(std::forward<Ts>(args)...); }
    case GDF_CATEGORY:  { return f.template operator()< category  >(std::forward<Ts>(args)...); }
//    case GDF_CATEGORY:  { return invoke_if_possible<T>{}(f, std::forward<Ts>(args)...); }
    default:            { assert(false && "type_dispatcher: invalid gdf_type"); }
  }
  return fallback_return_type();
}

} // namespace cudf

#endif
