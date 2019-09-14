/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "GTest.hpp"
#include "TypeList.hpp"

using namespace util; // this well make reading code way easier

namespace std {
template <class T, class U>
constexpr bool is_same_v = std::is_same<T, U>::value;
}

// GetType -----------------------------------------------------
static_assert(std::is_same<GetType<int, 0>, int>::value, "");
static_assert(std::is_same<GetType<Types<int, float, double>, 0>, int>::value, "");
static_assert(std::is_same<GetType<Types<int, float, double>, 1>, float>::value, "");
static_assert(std::is_same<GetType<Types<int, float, double>, 2>, double>::value, "");

// Our tests above will run at compile time.
// Let's just create a dummy test here to appear
// in the test run output. This will make us sure that
// this file was compiled.
TEST(TypeList, GetType) {}

// Concat -----------------------------------------------------
static_assert(std::is_same<Concat<>, Types<>>::value, "");

static_assert(std::is_same<Concat<Types<long, void *, char *>>, Types<long, void *, char *>>::value, "");

static_assert(std::is_same<Concat<Types<long, void *, char *>, Types<float, char, double>>,
                           Types<long, void *, char *, float, char, double>>::value,
              "");

static_assert(
    std::is_same<Concat<Types<long, void *, char *>, Types<float, char, double>, Types<int *, long *, unsigned>>,
                 Types<long, void *, char *, float, char, double, int *, long *, unsigned>>::value,
    "");

TEST(TypeList, Concat) {}

// Flatten -----------------------------------------------------
static_assert(std::is_same<Flatten<Types<>>, Types<>>::value, "");
static_assert(std::is_same<Flatten<Types<int>>, Types<int>>::value, "");
static_assert(std::is_same<Flatten<Types<int, double>>, Types<int, double>>::value, "");
static_assert(std::is_same<Flatten<Types<Types<int, double>, float>>, Types<int, double, float>>::value, "");
static_assert(std::is_same<Flatten<Types<Types<int, Types<double>>, float>>, Types<int, double, float>>::value, "");

TEST(TypeList, Flatten) {}

// CrossJoin -----------------------------------------------------
static_assert(std::is_same<CrossJoin<>, Types<>>::value, "");
static_assert(std::is_same<CrossJoin<Types<>, Types<>>, Types<>>::value, "");
static_assert(std::is_same<CrossJoin<Types<>, Types<int>>, Types<>>::value, "");
static_assert(std::is_same<CrossJoin<Types<>, Types<int, double>>, Types<>>::value, "");

static_assert(std::is_same<CrossJoin<Types<>, Types<int, double>, Types<>>, Types<>>::value, "");

static_assert(std::is_same<CrossJoin<Types<>, Types<>, Types<>>, Types<>>::value, "");

static_assert(std::is_same<CrossJoin<Types<int, double>, Types<>, Types<>>, Types<>>::value, "");

static_assert(std::is_same<CrossJoin<Types<int>, Types<>>, Types<>>::value, "");
static_assert(std::is_same<CrossJoin<Types<int, double>, Types<>>, Types<>>::value, "");

static_assert(std::is_same<CrossJoin<Types<int>, Types<int>>, Types<Types<int, int>>>::value, "");

static_assert(
    std::is_same<CrossJoin<Types<int, double>, Types<int>>, Types<Types<int, int>, Types<double, int>>>::value, "");

static_assert(
    std::is_same<CrossJoin<Types<int>, Types<double, char>>, Types<Types<int, double>, Types<int, char>>>::value, "");
static_assert(
    std::is_same<CrossJoin<Types<int, double>, Types<short, char>>,
                 Types<Types<int, short>, Types<int, char>, Types<double, short>, Types<double, char>>>::value,
    "");

static_assert(std::is_same<CrossJoin<Types<int, double>, Types<int>, Types<float>>,
                           Types<Types<int, int, float>, Types<double, int, float>>>::value,
              "");

static_assert(std::is_same<CrossJoin<Types<int, double>, Types<int>, Types<float, char>>,
                           Types<Types<int, int, float>, Types<int, int, char>, Types<double, int, float>,
                                 Types<double, int, char>>>::value,
              "");

static_assert(std::is_same<CrossJoin<Types<int, double>, Types<int, short>, Types<float, char>>,
                           Types<Types<int, int, float>, Types<int, int, char>, Types<int, short, float>,
                                 Types<int, short, char>, Types<double, int, float>, Types<double, int, char>,
                                 Types<double, short, float>, Types<double, short, char>>>::value,
              "");

static_assert(std::is_same<CrossJoin<Types<int, float>, int>, Types<Types<int, int>, Types<float, int>>>::value, "");
static_assert(std::is_same<CrossJoin<int, Types<int, float>>, Types<Types<int, int>, Types<int, float>>>::value, "");

TEST(TypeList, CrossJoin) {}

// AllSame -----------------------------------------------------
static_assert(AllSame::Call<Types<int, int>>::value, "");
static_assert(!AllSame::Call<Types<bool, int>>::value, "");

static_assert(AllSame::Call<int, int>::value, "");
static_assert(!AllSame::Call<int, bool>::value, "");

static_assert(AllSame::Call<int, int, int>::value, "");
static_assert(!AllSame::Call<int, float, int>::value, "");
static_assert(!AllSame::Call<int, int, float>::value, "");

TEST(TypeList, AllSame) {}

/*
// Exists -----------------------------------------------------
static_assert(Exists<int, Types<int, char, float>>, "");
static_assert(!Exists<int, Types<double, char, float>>, "");
static_assert(!Exists<int, Types<>>, "");
static_assert(Exists<int, Types<double, char, float, int>>, "");
static_assert(!Exists<int, Types<double>>, "");
static_assert(Exists<int, Types<int>>, "");
*/

TEST(TypeList, Exists) {}

// ContainedIn -----------------------------------------------------
static_assert(ContainedIn<Types<Types<int, char>>>::Call<Types<int, char>>::value, "");
static_assert(!ContainedIn<Types<Types<int, char>>>::Call<Types<int, float>>::value, "");
static_assert(!ContainedIn<Types<>>::Call<Types<int, float>>::value, "");
static_assert(ContainedIn<Types<Types<int, float>, Types<char, char>>>::Call<Types<int, float>>::value, "");
static_assert(!ContainedIn<Types<Types<int, float>, Types<char, char>>>::Call<Types<int, double>>::value, "");
static_assert(ContainedIn<Types<Types<int, float>, Types<>>>::Call<Types<>>::value, "");
static_assert(!ContainedIn<Types<Types<int, float>, Types<int>>>::Call<Types<>>::value, "");

TEST(TypeList, ContainedIn) {}
/*

// RemoveIf -----------------------------------------------------
static_assert(std::is_same<RemoveIf<AllSame, Types<>>, Types<>>::value, "");

static_assert(std::is_same<RemoveIf<AllSame, Types<Types<int, int, int>>>, Types<>>::value, "");

static_assert(std::is_same<RemoveIf<AllSame, Types<Types<int, float, int>>>, Types<Types<int, float, int>>>::value, "");

static_assert(
    std::is_same<RemoveIf<AllSame, Types<Types<int, float, char>, Types<int, int, int>, Types<int, int, char>>>,
                 Types<Types<int, float, char>, Types<int, int, char>>>::value,
    "");

static_assert(
    std::is_same<RemoveIf<AllSame, Types<Types<int, float, char>, Types<int, float, char>, Types<int, int, char>>>,
                 Types<Types<int, float, char>, Types<int, float, char>, Types<int, int, char>>>::value,
    "");

static_assert(std::is_same_v<RemoveIf<ContainedIn<Types<Types<int, char>, Types<float, int>>>,
                                      Types<Types<char, char>, Types<float, int>, Types<int, int>>>,
                             Types<Types<char, char>, Types<int, int>>>,
              "");

TEST(TypeList, RemoveIn) {}

// Transform -----------------------------------------------------
static_assert(std::is_same_v<Transform<Rep<2>, Types<int, float>>, Types<Types<int, int>, Types<float, float>>>);
static_assert(std::is_same_v<Transform<Rep<1>, Types<int, float>>, Types<Types<int>, Types<float>>>);
static_assert(std::is_same_v<Transform<Rep<0>, Types<int, float>>, Types<Types<>, Types<>>>);
static_assert(std::is_same_v<Transform<Rep<2>, Types<int>>, Types<Types<int, int>>>);
static_assert(std::is_same_v<Transform<Rep<1>, Types<int>>, Types<Types<int>>>);
static_assert(std::is_same_v<Transform<Rep<0>, Types<>>, Types<>>);

TEST(TypeList, Transform) {}

// Append -----------------------------------------------------
static_assert(std::is_same_v<Append<Types<>>, Types<>>);
static_assert(std::is_same_v<Append<Types<>, int>, Types<int>>);
static_assert(std::is_same_v<Append<Types<int>>, Types<int>>);
static_assert(std::is_same_v<Append<Types<int>, float>, Types<int, float>>);
static_assert(std::is_same_v<Append<Types<int>, float, char>, Types<int, float, char>>);

TEST(TypeList, Append) {}

// Unique -----------------------------------------------------
static_assert(std::is_same_v<Unique<Types<>>, Types<>>);
static_assert(std::is_same_v<Unique<Types<int, char, float>>, Types<int, char, float>>);
static_assert(std::is_same_v<Unique<Types<int, int, float>>, Types<int, float>>);
static_assert(std::is_same_v<Unique<Types<int, char, char, float>>, Types<int, char, float>>);
static_assert(std::is_same_v<Unique<Types<int, char, float, float>>, Types<int, char, float>>);

TEST(TypeList, Unique) {}

// Contains -------------------------------------------------

static_assert(Contains(Types<Size2D<1, 3>>(), nv::vpi::priv::Size2D{1, 3}));
static_assert(Contains(Types<Size2D<5, 2>, Size2D<3, 5>, Size2D<1, 3>>(), nv::vpi::priv::Size2D{1, 3}));
static_assert(!Contains(Types<Size2D<5, 2>, Size2D<3, 5>, Size2D<1, 3>>(), nv::vpi::priv::Size2D{1, 7}));
static_assert(!Contains(Types<Size2D<1, 4>>(), nv::vpi::priv::Size2D{1, 3}));
static_assert(!Contains(Types<>(), nv::vpi::priv::Size2D{1, 3}));

TEST(TypeList, Contains) {}

// Remove ---------------------------------------------------

static_assert(std::is_same<Remove<Types<int, float, char>, 1>, Types<int, char>>::value);
static_assert(std::is_same<Remove<Types<int, float, char>, 0, 2>, Types<float>>::value);
static_assert(std::is_same<Remove<Types<int, char>>, Types<int, char>>::value);
static_assert(std::is_same<Remove<Types<int>, 0>, Types<>>::value);
static_assert(std::is_same<Remove<Types<int, char>, 0, 1>, Types<>>::value);
static_assert(std::is_same<Remove<Types<>>, Types<>>::value);

TEST(TypeList, Remove) {}

// GetSize ------------------------------------------------

static_assert(GetSize<Types<>> == 0);
static_assert(GetSize<Types<int>> == 1);
static_assert(GetSize<Types<int, int>> == 2);
static_assert(GetSize<Types<int, void>> == 2);

TEST(TypeList, GetSize) {}

*/