/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef NV_VPI_TEST_UTIL_TYPELIST_HPP
#define NV_VPI_TEST_UTIL_TYPELIST_HPP

#include <common/Compiler.hpp>
#include <common/Size2D.hpp>
#include "GTest.hpp"

namespace util {

// Utilities for creating parameters for typed tests on GoogleTest
// We support both typed and (constexpr) value parameters. In order to work them
// seamslessly, we wrap the value into type Value
//
// Types is used to define type list, it's just an alias to ::testing::Types:
// using Types = util::Types<int,char,float>;
//
// Values is used to define compile-time value lists
// using Values = util::Values<'c',3.0,100>;
//
// You can declare your tests using a template fixture:
//
// template <class T>
// class TestFixture : ::testing::Test
// {
// };
//
// Inside your test function, you can access the parameters like this:
//
// TEST(TestFixture, mytest)
// {
//      using MEM = GetType<TypeParam,0>; // the first type element
//      constexpr auto VALUE = GetValue<TypeParam,1>; // second value element
// }
//
// You can compose complicated type/value arguments using Concat and CrossJoin and RemoveIf
//
// using Types = CrossJoin<Types<int,float>,Values<0,4>>;
// creates the parameters <int,0> <int,4> <float,0> <float,4>
//
// Concat is useful to concatenate parameter lists created with CrossJoin
//
// using Types = Concat<CrossJoin<Types<int,float>,ValueType<0>>,
//                      CrossJoin<Types<char,double>,ValueType<1,2>>>;
// creates the parameters <int,0> <float,0> <char,1> <char,2> <double,1> <double,2>
//
// RemoveIf can be used to remove some parameters that match a given predicate:
//
// using Types = RemoveIf<AllSame, CrossJoin<Types<int,char>, Types<int,char>>>;
// creates the parameters <int,char>,<char,int>

// Types -----------------------------------------

using ::testing::Types;

template<class T, int D>
struct GetTypeImpl
{
    static_assert(D == 0, "Out of bounds");
    using type = T;
};

template<class... T, int D>
struct GetTypeImpl<Types<T...>, D>
{
    static_assert(D < sizeof...(T), "Out of bounds");

    using type = typename GetTypeImpl<typename Types<T...>::Tail, D - 1>::type;
};

template<class... ARGS>
struct GetTypeImpl<Types<ARGS...>, 0>
{
    static_assert(sizeof...(ARGS) > 0, "Out of bounds");

    using type = typename Types<ARGS...>::Head;
};

template<class TUPLE, int D>
using GetType = typename GetTypeImpl<TUPLE, D>::type;

// GetSize -------------------------------
// returns the size (number of elements) of the type list

template<class TUPLE>
struct GetSizeImpl;

template<class... TYPES>
struct GetSizeImpl<Types<TYPES...>>
{
    static constexpr auto value = sizeof...(TYPES);
};

template<class TUPLE>
constexpr auto GetSize = GetSizeImpl<TUPLE>::value;

// Values -----------------------------------------

// NOTE: be aware that gcc-7.x has a bug where V's type will be
// wrong if the value was already instantiated with another type.
// gcc-8.0 fixes it. clang-6.0.0 doesn't have this bug.
// decltype(GetValue<Values<1,3ul,1>>) == 'const int' instead of 'const unsigned long'
// Ref: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79092
// Our WAR involves defining Value without using auto, i.e.
// Value<class T, T V>, and creating a macro that uses decltype
// to define 'class T', so that we don't have to write the type
// ourselves.

#if VPI_GCC_VERSION >= 80000 || VPI_CLANG_VERSION >= 60000
template<auto V>
struct ValueImpl
{
    static constexpr auto value = V;
};

template<auto... ARGS>
struct ValuesImpl
{
    using type = Types<ValueImpl<ARGS>...>;
};

#define Value(V) ValueImpl<V>

#else

template<class T, T V>
struct ValueImpl
{
    static constexpr auto value = V;
};

#define Value(V) ValueImpl<decltype(V), V>

template<auto... ARGS>
struct ValuesImpl
{
    using type = Types<Value(ARGS)...>;
};

#endif

template<auto... ARGS>
using Values = typename ValuesImpl<ARGS...>::type;

template<class TUPLE, int D>
constexpr auto GetValue = GetType<TUPLE, D>::value;

// Concat -----------------------------------------

namespace detail {
template<class A, class B>
struct Concat2;

template<class... T, class... U>
struct Concat2<Types<T...>, Types<U...>>
{
    using type = Types<T..., U...>;
};
} // namespace detail

template<class... T>
struct ConcatImpl;

template<class HEAD1, class HEAD2, class... TAIL>
struct ConcatImpl<HEAD1, HEAD2, TAIL...>
{
    using type = typename ConcatImpl<typename detail::Concat2<HEAD1, HEAD2>::type, TAIL...>::type;
};

template<class A>
struct ConcatImpl<A>
{
    using type = A;
};

template<class... A>
struct ConcatImpl<Types<A...>>
{
    using type = Types<A...>;
};

template<>
struct ConcatImpl<>
{
    using type = Types<>;
};

template<class... T>
using Concat = typename ConcatImpl<T...>::type;

// Flatten -----------------------------------------

template<class T>
struct FlattenImpl;

template<>
struct FlattenImpl<Types<>>
{
    using type = Types<>;
};

template<class HEAD, class... TAIL>
struct FlattenImpl<Types<HEAD, TAIL...>>
{
    using type = Concat<Types<HEAD>, typename FlattenImpl<Types<TAIL...>>::type>;
};

template<class... HEAD, class... TAIL>
struct FlattenImpl<Types<Types<HEAD...>, TAIL...>>
{
    using type = typename FlattenImpl<Types<HEAD..., TAIL...>>::type;
};

template<class T>
using Flatten = typename FlattenImpl<T>::type;

// CrossJoin -----------------------------------------

namespace detail {
// prepend T in TUPLE
template<class T, class TUPLE>
struct Prepend1;

template<class T, class... ARGS>
struct Prepend1<T, Types<ARGS...>>
{
    using type = Flatten<Types<T, ARGS...>>;
};

template<class T, class TUPLES>
struct Prepend;

// Prepend T in all TUPLES
template<class T, class... TUPLES>
struct Prepend<T, Types<TUPLES...>>
{
    using type = Types<typename Prepend1<T, TUPLES>::type...>;
};

// skip empty tuples
template<class T, class... TUPLES>
struct Prepend<T, Types<Types<>, TUPLES...>> : Prepend<T, Types<TUPLES...>>
{
};
} // namespace detail

template<class... ARGS>
struct CrossJoinImpl;

template<>
struct CrossJoinImpl<>
{
    using type = Types<>;
};

template<class... ARGS>
struct CrossJoinImpl<Types<ARGS...>>
{
    using type = Types<Types<ARGS>...>;
};

template<class... AARGS, class... TAIL>
struct CrossJoinImpl<Types<AARGS...>, TAIL...>
{
    using type = Concat<typename detail::Prepend<AARGS, typename CrossJoinImpl<TAIL...>::type>::type...>;
};

// to make it easy for the user when there's only one element to be joined
template<class T, class... TAIL>
struct CrossJoinImpl<T, TAIL...> : CrossJoinImpl<Types<T>, TAIL...>
{
};

template<class... ARGS>
using CrossJoin = typename CrossJoinImpl<ARGS...>::type;

// AllSame -----------------------------------------

namespace detail {
template<class... ITEMS>
struct AllSame : std::false_type
{
};

// degenerate case
template<class A>
struct AllSame<A> : std::true_type
{
};

template<class A>
struct AllSame<A, A> : std::true_type
{
};

template<class HEAD, class... TAIL>
struct AllSame<HEAD, HEAD, TAIL...> : AllSame<HEAD, TAIL...>
{
};

template<class... ITEMS>
struct AllSame<Types<ITEMS...>> : AllSame<ITEMS...>
{
};

} // namespace detail

struct AllSame
{
    template<class... ITEMS>
    using Call = detail::AllSame<ITEMS...>;
};

// Exists ---------------------------------

// Do a linear search to find NEEDLE in HAYSACK
template<class NEEDLE, class HAYSACK>
struct ExistsImpl;

// end case, no more types to check
template<class NEEDLE>
struct ExistsImpl<NEEDLE, Types<>> : std::false_type
{
};

// next one matches
template<class NEEDLE, class... TAIL>
struct ExistsImpl<NEEDLE, Types<NEEDLE, TAIL...>> : std::true_type
{
};

// next one doesn't match
template<class NEEDLE, class HEAD, class... TAIL>
struct ExistsImpl<NEEDLE, Types<HEAD, TAIL...>> : ExistsImpl<NEEDLE, Types<TAIL...>>
{
};

template<class NEEDLE, class HAYSACK>
constexpr bool Exists = ExistsImpl<NEEDLE, HAYSACK>::value;

// ContainedIn -----------------------------------------

template<class HAYSACK>
struct ContainedIn
{
    template<class NEEDLE>
    using Call = ExistsImpl<NEEDLE, HAYSACK>;
};

// RemoveIf -----------------------------------------

template<class PRED, class TUPLE>
struct RemoveIfImpl;

template<class PRED>
struct RemoveIfImpl<PRED, Types<>>
{
    using type = Types<>;
};

template<class PRED, class HEAD, class... TAIL>
struct RemoveIfImpl<PRED, Types<HEAD, TAIL...>>
{
    using type = Concat<typename std::conditional<PRED::template Call<HEAD>::value, Types<>, Types<HEAD>>::type,
                        typename RemoveIfImpl<PRED, Types<TAIL...>>::type>;
};

template<class PRED, class TUPLE>
using RemoveIf = typename RemoveIfImpl<PRED, TUPLE>::type;

// Transform --------------------------------

template<class XFORM, class TYPES>
struct TransformImpl;

template<class XFORM, class... ITEMS>
struct TransformImpl<XFORM, Types<ITEMS...>>
{
    using type = Types<typename XFORM::template Call<ITEMS>...>;
};

template<class XFORM, class TYPES>
using Transform = typename TransformImpl<XFORM, TYPES>::type;

// Rep --------------------------------

namespace detail {
template<class T, int N, class RES>
struct Rep;

template<class T, int N, class... ITEMS>
struct Rep<T, N, Types<ITEMS...>>
{
    using type = typename Rep<T, N - 1, Types<T, ITEMS...>>::type;
};

template<class T, class... ITEMS>
struct Rep<T, 0, Types<ITEMS...>>
{
    using type = Types<ITEMS...>;
};
} // namespace detail

template<int N>
struct Rep
{
    template<class T>
    using Call = typename detail::Rep<T, N, Types<>>::type;
};

// Append --------------------------------

template<class TYPES, class... ITEMS>
struct AppendImpl;

template<class... HEAD, class... TAIL>
struct AppendImpl<Types<HEAD...>, TAIL...>
{
    using type = Types<HEAD..., TAIL...>;
};

template<class TYPES, class... ITEMS>
using Append = typename AppendImpl<TYPES, ITEMS...>::type;

// Remove -------------------------------------------
// remove items from tuple given by their indices

namespace detail {
template<class TUPLE, int CUR, int... IDXs>
struct Remove;

// nothing else to do?
template<class... ITEMS, int CUR>
struct Remove<Types<ITEMS...>, CUR>
{
    using type = Types<ITEMS...>;
};

// index match current item?
template<class HEAD, class... TAIL, int CUR, int... IDXTAIL>
struct Remove<Types<HEAD, TAIL...>, CUR, CUR, IDXTAIL...>
{
    // remove it, and recurse into the remaining items
    using type = typename Remove<Types<TAIL...>, CUR + 1, IDXTAIL...>::type;
};

// index doesn't match current item?
template<class HEAD, class... TAIL, int CUR, int IDXHEAD, int... IDXTAIL>
struct Remove<Types<HEAD, TAIL...>, CUR, IDXHEAD, IDXTAIL...>
{
    static_assert(sizeof...(TAIL) + 1 > IDXHEAD - CUR, "Index out of bounds");

    // add current item to output and recurse into the remaining items
    using type = Concat<Types<HEAD>, typename Remove<Types<TAIL...>, CUR + 1, IDXHEAD, IDXTAIL...>::type>;
};
} // namespace detail

template<class TUPLE, int... IDXs>
struct RemoveImpl
{
    using type = typename detail::Remove<TUPLE, 0, IDXs...>::type;
};

template<class TUPLE, int... IDXs>
using Remove = typename RemoveImpl<TUPLE, IDXs...>::type;

// Unique --------------------------------

namespace detail {
template<class... ITEMS>
struct Unique;

template<>
struct Unique<>
{
    using type = Types<>;
};

template<class HEAD, class... TAIL>
struct Unique<HEAD, TAIL...>
{
    using type =
        Concat<std::conditional_t<Exists<HEAD, Types<TAIL...>>, Types<>, Types<HEAD>>, typename Unique<TAIL...>::type>;
};
} // namespace detail

template<class TYPES>
struct UniqueImpl;

template<class... ITEMS>
struct UniqueImpl<Types<ITEMS...>>
{
    using type = typename detail::Unique<ITEMS...>::type;
};

template<class TYPES>
using Unique = typename UniqueImpl<TYPES>::type;

// Helper to be able to define typed test cases by defining the
// types inline (no worries about commas in macros)

#define VPI_TYPED_TEST_SUITE_F(TEST, ...) \
    using TEST##_Types = __VA_ARGS__;     \
    TYPED_TEST_SUITE(TEST, TEST##_Types)

#define VPI_TYPED_TEST_SUITE(TEST, ...) \
    template<class T>                   \
    class TEST : public ::testing::Test \
    {                                   \
    };                                  \
    VPI_TYPED_TEST_SUITE_F(TEST, __VA_ARGS__)

#define VPI_INSTANTIATE_TYPED_TEST_SUITE_P(INSTNAME, TEST, ...) \
    using TEST##INSTNAME##_Types = __VA_ARGS__;                 \
    INSTANTIATE_TYPED_TEST_SUITE_P(INSTNAME, TEST, TEST##INSTNAME##_Types)

// With C++17 we still can't have POD as non-type template parameters,
// so we need this little helper to create a type argument list of
// sizes. C++20 will support it though.
template<int W, int H>
struct Size2D
{
    static constexpr nv::vpi::priv::Size2D value = nv::vpi::priv::Size2D{W, H};
};

// Contains ------------------------------
// check if value is in util::Values container

template<class T>
constexpr bool Contains(Types<>, T size)
{
    return false;
}

template<int HEAD_W, int... TAIL_W, int HEAD_H, int... TAIL_H>
constexpr bool Contains(Types<Size2D<HEAD_W, HEAD_H>, Size2D<TAIL_W, TAIL_H>...>, nv::vpi::priv::Size2D size)
{
    if (HEAD_W == size.w && HEAD_H == size.h)
    {
        return true;
    }
    else
    {
        return Contains(Types<Size2D<TAIL_W, TAIL_H>...>(), size);
    }
}

template<class T, auto HEAD, auto... TAIL>
constexpr bool Contains(Types<Value(HEAD), Value(TAIL)...>, T needle)
{
    if (HEAD == needle)
    {
        return true;
    }
    else
    {
        return Contains(Types<Value(TAIL)...>(), needle);
    }
}

} // namespace util

#endif // NV_VPI_TEST_UTIL_TYPELIST_HPP
