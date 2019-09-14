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

using namespace util;  // this will make reading code way easier

namespace std {
template <class T, class U>
constexpr bool is_same_v = std::is_same<T, U>::value;
}

namespace {
// Work around to remove paranthesis surrounding a type
template <typename T>
struct argument_type;
template <typename T, typename U>
struct argument_type<T(U)> {
  using type = U;
};
}  // namespace
/**---------------------------------------------------------------------------*
 * @brief Performs a compile-time check that two types are equivalent.
 *
 * @note In order to work around commas in macros, any type containing commas
 * should be wrapped in paranthesis.
 *
 * Example:
 * ```
 * EXPECT_SAME_TYPE(int, int);
 *
 * EXPECT_SAME_TYPE(int, float); // compile error
 *
 * // Paranthesis around types with commas
 * EXPECT_SAME_TYPE((std::map<int, float), (std::map<int, float>));
 * ```
 *---------------------------------------------------------------------------**/
#define EXPECT_SAME_TYPE(expected, actual)                          \
  static_assert(std::is_same_v<argument_type<void(expected)>::type, \
                               argument_type<void(actual)>::type>,  \
                "");

TEST(TypeList, GetType) {
  EXPECT_SAME_TYPE((GetType<int, 0>), int);
  EXPECT_SAME_TYPE((GetType<Types<int, float, double>, 0>), int);
  EXPECT_SAME_TYPE((GetType<Types<int, float, double>, 1>), float);
  EXPECT_SAME_TYPE((GetType<Types<int, float, double>, 2>), double);
}

TEST(TypeList, Concat) {
  EXPECT_SAME_TYPE(Concat<>, Types<>);
  EXPECT_SAME_TYPE((Concat<Types<long, void *, char *>>),
                   (Types<long, void *, char *>));

  EXPECT_SAME_TYPE(
      (Concat<Types<long, void *, char *>, Types<float, char, double>>),
      (Types<long, void *, char *, float, char, double>));

  EXPECT_SAME_TYPE(
      (Concat<Types<long, void *, char *>, Types<float, char, double>,
              Types<int *, long *, unsigned>>),
      (Types<long, void *, char *, float, char, double, int *, long *,
             unsigned>));
}

TEST(TypeList, Flatten) {
  EXPECT_SAME_TYPE(Flatten<Types<>>, Types<>);
  EXPECT_SAME_TYPE(Flatten<Types<int>>, Types<int>);
  EXPECT_SAME_TYPE((Flatten<Types<int, double>>), (Types<int, double>));
  EXPECT_SAME_TYPE((Flatten<Types<Types<int, double>, float>>),
                   (Types<int, double, float>));
  EXPECT_SAME_TYPE((Flatten<Types<Types<int, Types<double>>, float>>),
                   (Types<int, double, float>));
}

TEST(TypeList, CrossJoin) {
  EXPECT_SAME_TYPE(CrossJoin<>, Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<>, Types<int>>), Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<>, Types<int, double>>), Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<>, Types<int, double>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<>, Types<>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<int, double>, Types<>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<int>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<int, double>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossJoin<Types<int>, Types<int>>),
                   (Types<Types<int, int>>));
  EXPECT_SAME_TYPE((CrossJoin<Types<int, double>, Types<int>>),
                   (Types<Types<int, int>, Types<double, int>>));
  EXPECT_SAME_TYPE((CrossJoin<Types<int>, Types<double, char>>),
                   (Types<Types<int, double>, Types<int, char>>));
  EXPECT_SAME_TYPE((CrossJoin<Types<int, double>, Types<short, char>>),
                   (Types<Types<int, short>, Types<int, char>,
                          Types<double, short>, Types<double, char>>));
  EXPECT_SAME_TYPE((CrossJoin<Types<int, double>, Types<int>, Types<float>>),
                   (Types<Types<int, int, float>, Types<double, int, float>>));

  EXPECT_SAME_TYPE(
      (CrossJoin<Types<int, double>, Types<int>, Types<float, char>>),
      (Types<Types<int, int, float>, Types<int, int, char>,
             Types<double, int, float>, Types<double, int, char>>));
  EXPECT_SAME_TYPE(
      (CrossJoin<Types<int, double>, Types<int, short>, Types<float, char>>),
      (Types<Types<int, int, float>, Types<int, int, char>,
             Types<int, short, float>, Types<int, short, char>,
             Types<double, int, float>, Types<double, int, char>,
             Types<double, short, float>, Types<double, short, char>>));
  EXPECT_SAME_TYPE((CrossJoin<Types<int, float>, int>),
                   (Types<Types<int, int>, Types<float, int>>));
  EXPECT_SAME_TYPE((CrossJoin<int, Types<int, float>>),
                   (Types<Types<int, int>, Types<int, float>>));
}

TEST(TypeList, AllSame) {
  static_assert(AllSame::Call<Types<int, int>>::value, "");
  static_assert(AllSame::Call<Types<int, int>>::value, "");
  static_assert(!AllSame::Call<Types<bool, int>>::value, "");

  static_assert(AllSame::Call<int, int>::value, "");
  static_assert(!AllSame::Call<int, bool>::value, "");

  static_assert(AllSame::Call<int, int, int>::value, "");
  static_assert(!AllSame::Call<int, float, int>::value, "");
  static_assert(!AllSame::Call<int, int, float>::value, "");
}

TEST(TypeList, Exists) {
  static_assert(Exists<int, Types<int, char, float>>, "");
  static_assert(!Exists<int, Types<double, char, float>>, "");
  static_assert(!Exists<int, Types<>>, "");
  static_assert(Exists<int, Types<double, char, float, int>>, "");
  static_assert(!Exists<int, Types<double>>, "");
  static_assert(Exists<int, Types<int>>, "");
}

TEST(TypeList, ContainedIn) {
  static_assert(
      ContainedIn<Types<Types<int, char>>>::Call<Types<int, char>>::value, "");
  static_assert(
      !ContainedIn<Types<Types<int, char>>>::Call<Types<int, float>>::value,
      "");
  static_assert(!ContainedIn<Types<>>::Call<Types<int, float>>::value, "");
  static_assert(ContainedIn<Types<Types<int, float>, Types<char, char>>>::Call<
                    Types<int, float>>::value,
                "");
  static_assert(!ContainedIn<Types<Types<int, float>, Types<char, char>>>::Call<
                    Types<int, double>>::value,
                "");
  static_assert(
      ContainedIn<Types<Types<int, float>, Types<>>>::Call<Types<>>::value, "");
  static_assert(
      !ContainedIn<Types<Types<int, float>, Types<int>>>::Call<Types<>>::value,
      "");
}

TEST(TypeList, RemoveIf) {
  EXPECT_SAME_TYPE((RemoveIf<AllSame, Types<>>), Types<>);

  EXPECT_SAME_TYPE((RemoveIf<AllSame, Types<Types<int, int, int>>>), Types<>);

  EXPECT_SAME_TYPE((RemoveIf<AllSame, Types<Types<int, float, int>>>),
                   (Types<Types<int, float, int>>));

  EXPECT_SAME_TYPE(
      (RemoveIf<AllSame, Types<Types<int, float, char>, Types<int, int, int>,
                               Types<int, int, char>>>),
      (Types<Types<int, float, char>, Types<int, int, char>>));

  EXPECT_SAME_TYPE(
      (RemoveIf<AllSame, Types<Types<int, float, char>, Types<int, float, char>,
                               Types<int, int, char>>>),
      (Types<Types<int, float, char>, Types<int, float, char>,
             Types<int, int, char>>));

  EXPECT_SAME_TYPE(
      (RemoveIf<ContainedIn<Types<Types<int, char>, Types<float, int>>>,
                Types<Types<char, char>, Types<float, int>, Types<int, int>>>),
      (Types<Types<char, char>, Types<int, int>>));
}

/*
// Transform -----------------------------------------------------
static_assert(std::is_same_v<Transform<Rep<2>, Types<int, float>>,
Types<Types<int, int>, Types<float, float>>>);
static_assert(std::is_same_v<Transform<Rep<1>, Types<int, float>>,
Types<Types<int>, Types<float>>>);
static_assert(std::is_same_v<Transform<Rep<0>, Types<int, float>>,
Types<Types<>, Types<>>>); static_assert(std::is_same_v<Transform<Rep<2>,
Types<int>>, Types<Types<int, int>>>);
static_assert(std::is_same_v<Transform<Rep<1>, Types<int>>, Types<Types<int>>>);
static_assert(std::is_same_v<Transform<Rep<0>, Types<>>, Types<>>);

TEST(TypeList, Transform) {}

// Append -----------------------------------------------------
static_assert(std::is_same_v<Append<Types<>>, Types<>>);
static_assert(std::is_same_v<Append<Types<>, int>, Types<int>>);
static_assert(std::is_same_v<Append<Types<int>>, Types<int>>);
static_assert(std::is_same_v<Append<Types<int>, float>, Types<int, float>>);
static_assert(std::is_same_v<Append<Types<int>, float, char>, Types<int, float,
char>>);

TEST(TypeList, Append) {}

// Unique -----------------------------------------------------
static_assert(std::is_same_v<Unique<Types<>>, Types<>>);
static_assert(std::is_same_v<Unique<Types<int, char, float>>, Types<int, char,
float>>); static_assert(std::is_same_v<Unique<Types<int, int, float>>,
Types<int, float>>); static_assert(std::is_same_v<Unique<Types<int, char, char,
float>>, Types<int, char, float>>);
static_assert(std::is_same_v<Unique<Types<int, char, float, float>>, Types<int,
char, float>>);

TEST(TypeList, Unique) {}

// Contains -------------------------------------------------

static_assert(Contains(Types<Size2D<1, 3>>(), nv::vpi::priv::Size2D{1, 3}));
static_assert(Contains(Types<Size2D<5, 2>, Size2D<3, 5>, Size2D<1, 3>>(),
nv::vpi::priv::Size2D{1, 3})); static_assert(!Contains(Types<Size2D<5, 2>,
Size2D<3, 5>, Size2D<1, 3>>(), nv::vpi::priv::Size2D{1, 7}));
static_assert(!Contains(Types<Size2D<1, 4>>(), nv::vpi::priv::Size2D{1, 3}));
static_assert(!Contains(Types<>(), nv::vpi::priv::Size2D{1, 3}));

TEST(TypeList, Contains) {}

// Remove ---------------------------------------------------

static_assert(std::is_same<Remove<Types<int, float, char>, 1>, Types<int,
char>>::value); static_assert(std::is_same<Remove<Types<int, float, char>, 0,
2>, Types<float>>::value); static_assert(std::is_same<Remove<Types<int, char>>,
Types<int, char>>::value); static_assert(std::is_same<Remove<Types<int>, 0>,
Types<>>::value); static_assert(std::is_same<Remove<Types<int, char>, 0, 1>,
Types<>>::value); static_assert(std::is_same<Remove<Types<>>, Types<>>::value);

TEST(TypeList, Remove) {}

// GetSize ------------------------------------------------

static_assert(GetSize<Types<>> == 0);
static_assert(GetSize<Types<int>> == 1);
static_assert(GetSize<Types<int, int>> == 2);
static_assert(GetSize<Types<int, void>> == 2);

TEST(TypeList, GetSize) {}

*/