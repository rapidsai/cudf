/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#ifndef TYPE_DISPATCHER_HPP
#define TYPE_DISPATCHER_HPP

#include <utilities/wrapper_types.hpp>
#include <utilities/release_assert.cuh>

#include <cudf/types.h>

// Forward decl
class NVStrings;

#include <utility>
#include <type_traits>

namespace cudf {

namespace detail {

template <typename... List> struct list { };

template < typename Tp, typename... List >
struct first_non_matching;

template <typename T> struct non_match { using type = T; };

template < typename Tp, typename Head, typename... Rest >
struct first_non_matching<Tp, Head, Rest...> :
    std::conditional<not std::is_same<Tp, Head>::value, non_match<Head>, first_non_matching<Tp, Rest...> >::type
{};

template < typename Tp >
struct first_non_matching<Tp> { }; // no type set

using all_column_element_types =
    list<
        int8_t, int16_t, int32_t, int64_t, float, double, cudf::date32, cudf::date64,
        cudf::timestamp, cudf::category, cudf::nvstring_category
    >;


template <typename>
struct sfinae_true : std::true_type{};

// Note: The .template is known as a "template disambiguator"
// See here for more information: https://stackoverflow.com/questions/3786360/confusing-template-error
template <class F, typename T, typename... Us>
static auto test_templated_invoke_operator(int) ->
    sfinae_true<decltype(std::declval<F>().template operator()<T>(std::declval<Us>()... ))>;

template <class, typename, typename... Us>
static auto test_templated_invoke_operator(long) -> std::false_type;

// A useful trait in itself:
template <class F, typename T, typename... Us>
struct has_templated_invoke_operator : decltype( test_templated_invoke_operator<F, T, Us...>(int{}) )
{ };

template <bool ActuallyInvoke, typename R, class F, typename T, typename... Ts>
struct invoke_if_possible_inner;

template <class F, typename R, typename T, typename... Ts>
struct invoke_if_possible_inner<false, R, F, T, Ts...>
{
    constexpr __host__ __device__
    R operator()(F functor, Ts&&... params) {
        // throw std::invalid_argument("Unsupported column element type");
        return R();
    }
};

template <class F, typename R, typename T, typename... Ts>
struct invoke_if_possible_inner<true, R, F, T, Ts...>
{
// Note that some functors may only have their operator()'s defined for host-side code,
// and other only for device-side code; but - since these decorators are not really part of
// a function's C++ type, we can't quite work our way around this issue, and have to
// result to the following kludge:
#pragma hd_warning_disable
    constexpr CUDA_HOST_DEVICE_CALLABLE
    R operator()(F functor, Ts&&... params)
    {
        return functor.template operator()<T>(std::forward<Ts>(params)...);
    }
};

template <typename T, typename R>
struct invoke_if_possible {
    template <class F, typename... Ts>
    constexpr CUDA_HOST_DEVICE_CALLABLE
    R operator()(F functor, Ts&&... params)
    {
        constexpr bool actually_invoke = has_templated_invoke_operator<F, T, Ts...>::value;
        static_assert(actually_invoke == true,"Should be able to invoke for now!");
        return invoke_if_possible_inner<actually_invoke, R, F, T, Ts...>{}(functor, std::forward<Ts>(params)...);
    }
};

struct no_invocation_operator_return_type {};

template <class F, typename T, bool HaveOperator, typename... Us>
struct invocation_operator_return_type_inner;

template <class F, typename T, typename... Us>
struct invocation_operator_return_type_inner<F, T, false, Us...> {
    using type = no_invocation_operator_return_type;
};

template <class F, typename T, typename... Us>
struct invocation_operator_return_type_inner<F, T, true, Us...> {
    using type = decltype(std::declval<F>().template operator()<T>(std::declval<Us>()... ));
};

template <class F, typename T, typename... Us>
using invocation_operator_return_type =
    typename invocation_operator_return_type_inner<F, T, has_templated_invoke_operator<F, T, Us...>::value, Us...>::type;

template <class F, typename... Us>
struct fnm_helper {

    template <typename T>
    using iort = invocation_operator_return_type<F, T, Us...>;

    template <typename... List>
    using transformed = typename first_non_matching<
        no_invocation_operator_return_type,
        iort<List>...>::type;
};

} // namespace detail

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
// Finally - Note that some or most of this ugly TMP can easily go away if C++17 or C++20 is available to
// us - using if constexpr to avoid SFINAE, and eventually using variadic lambdas which may eliminate
// the type_dispatcher altogether. Still, even with C++14 it's possible the

/**---------------------------------------------------------------------------*
 * @brief Maps a C++ type to it's corresponding gdf_dtype.
 *
 * When explicitly passed a template argument of a given type, returns the
 * appropriate `gdf_dtype` for the specified C++ type.
 *
 * For example:
 * ```
 * return gdf_dtype_of<int32_t>();        // Returns GDF_INT32
 * return gdf_dtype_of<cudf::category>(); // Returns GDF_CATEGORY
 * ```
 *
 * @tparam T The type to map to a `gdf_dtype`
 *---------------------------------------------------------------------------**/
template <typename T>
inline constexpr gdf_dtype gdf_dtype_of() {  return GDF_invalid; };

template <> inline constexpr gdf_dtype gdf_dtype_of< int8_t            >() { return GDF_INT8;            };
template <> inline constexpr gdf_dtype gdf_dtype_of< int16_t           >() { return GDF_INT16;           };
template <> inline constexpr gdf_dtype gdf_dtype_of< int32_t           >() { return GDF_INT32;           };
template <> inline constexpr gdf_dtype gdf_dtype_of< int64_t           >() { return GDF_INT64;           };
template <> inline constexpr gdf_dtype gdf_dtype_of< float             >() { return GDF_FLOAT32;         };
template <> inline constexpr gdf_dtype gdf_dtype_of< double            >() { return GDF_FLOAT64;         };
template <> inline constexpr gdf_dtype gdf_dtype_of< date32            >() { return GDF_DATE32;          };
template <> inline constexpr gdf_dtype gdf_dtype_of< date64            >() { return GDF_DATE64;          };
template <> inline constexpr gdf_dtype gdf_dtype_of< timestamp         >() { return GDF_TIMESTAMP;       };
template <> inline constexpr gdf_dtype gdf_dtype_of< category          >() { return GDF_CATEGORY;        };
template <> inline constexpr gdf_dtype gdf_dtype_of< nvstring_category >() { return GDF_STRING_CATEGORY; };
template <> inline constexpr gdf_dtype gdf_dtype_of< NVStrings         >() { return GDF_STRING;          };


/* --------------------------------------------------------------------------*/
/** 
 * @brief  Invokes an instance of a functor template with the appropriate type
 * determined by a gdf_dtype enum value.
 *
 * This helper function accepts any object with an "operator()" template,
 * e.g., a functor. It will invoke an instance of the template by passing
 * in as the template argument an appropriate type determined by the value of
 * the gdf_dtype argument.
 *
 * There is a 1-to-1 mapping of gdf_dtype enum values and dispatched types.
 * However, different gdf_dtype values may have the same underlying type.
 * Therefore, in order to provide the 1-to-1 mapping, a wrapper struct may be
 * dispatched for certain gdf_dtype enum values in order to emulate a "strong
 * typedef".
 *
 * A strong typedef  provides a new, concrete type unlike a normal C++ typedef which
 * is simply a type alias. These "strong typedef" structs simply wrap a single member
 * variable of a fundamental type called 'value'.
 *
 * The standard arithmetic operators are defined for the wrapper structs and
 * therefore the wrapper struct types can be used as if they were fundamental
 * types.
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
 * @note It is undefined behavior if an unsupported or invalid `gdf_dtype` is
 * supplied.
 * @note The function @p f does not need to have operator()<T> defined for all
 * column element types - just the types with which it is used. However, it
 * must have at least
 *
 * @param dtype The gdf_dtype enum that determines which type will be dispatched
 * @param f The functor with a templated "operator()" that will be invoked with 
 * the dispatched type
 * @param params A parameter-pack (i.e., arbitrary number of arguments) that will
 * be perfectly-forwarded as the arguments of the functor's "operator()".
 *
 * @returns Whatever is returned by the functor's "operator()".
 *
 */
/* ----------------------------------------------------------------------------*/

template < class functor_t, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE
constexpr auto type_dispatcher(
    gdf_dtype  dtype,
    functor_t  f, 
    Ts&&...    params)
{
  // TODO: It may be the case that since the operator() template must be at least
  // declared, even if it isn't defined, that we can get the return type without
  // resorting to the template metaprogramming voodoo below.
  //
  // This is how we would do it:
  //
  // using return_type = decltype(f.template operator()<int8_t>(std::forward<Ts>(params)...));

  using fnm_helper = typename detail::fnm_helper<functor_t, Ts...>;
  using return_type = typename fnm_helper::template transformed<
          int8_t, int16_t, int32_t, int64_t, float, double, cudf::date32, cudf::date64,
          cudf::timestamp, cudf::category, cudf::nvstring_category
      >;
  static_assert(not std::is_same<return_type, detail::no_invocation_operator_return_type>::value,
      "No appropriate operator() defined for _any_ column element data type");

  switch(dtype)
  {
    case gdf_dtype_of< int8_t            >(): { return detail::invoke_if_possible< int8_t           , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< int16_t           >(): { return detail::invoke_if_possible< int16_t          , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< int32_t           >(): { return detail::invoke_if_possible< int32_t          , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< int64_t           >(): { return detail::invoke_if_possible< int64_t          , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< float             >(): { return detail::invoke_if_possible< float            , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< double            >(): { return detail::invoke_if_possible< double           , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< date32            >(): { return detail::invoke_if_possible< date32           , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< date64            >(): { return detail::invoke_if_possible< date64           , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< timestamp         >(): { return detail::invoke_if_possible< timestamp        , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< category          >(): { return detail::invoke_if_possible< category         , return_type>{}(f,std::forward<Ts>(params)...); }
    case gdf_dtype_of< nvstring_category >(): { return detail::invoke_if_possible< nvstring_category, return_type>{}(f,std::forward<Ts>(params)...); }
    default: {
#ifdef __CUDA_ARCH__
      
      // This will cause the calling kernel to crash as well as invalidate
      // the GPU context
      release_assert(false && "Invalid gdf_dtype in type_dispatcher");

      // The following code will never be reached, but the compiler generates a
      // warning if there isn't a return value.

      // Need to find out what the return type is in order to have a default
      // return value and solve the compiler warning for lack of a default
      // return
      return return_type();
#else
      // In host-code, the compiler is smart enough to know we don't need a
      // default return type since we're throwing an exception.
      throw std::invalid_argument("Invalid gdf_dtype in type_dispatcher");
#endif
    }
  }
}


}  // namespace cudf

#endif
