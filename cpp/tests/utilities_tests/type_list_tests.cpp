/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf_test/type_list_utilities.hpp>

using namespace cudf::test;  // this will make reading code way easier

namespace {
// Work around to remove parentheses surrounding a type
template <typename T>
struct argument_type;

template <typename T, typename U>
struct argument_type<T(U)> {
  using type = U;
};
}  // namespace
/**
 * @brief Performs a compile-time check that two types are equivalent.
 *
 * @note In order to work around commas in macros, any type containing commas
 * should be wrapped in parentheses.
 *
 * Example:
 * ```
 * EXPECT_SAME_TYPE(int, int);
 *
 * EXPECT_SAME_TYPE(int, float); // compile error
 *
 * // Parentheses around types with commas
 * EXPECT_SAME_TYPE((std::map<int, float>), (std::map<int, float>));
 * ```
 */
#define EXPECT_SAME_TYPE(expected, actual) \
  static_assert(                           \
    std::is_same_v<argument_type<void(expected)>::type, argument_type<void(actual)>::type>, "");

/**
 * @brief Return a string of the demangled name of a type `T`
 *
 * This is useful for debugging Type list utilities.
 *
 * @tparam T The type whose name is returned as a string
 * @return std::string The demangled name of `T`
 */
template <typename T>
std::string type_name()
{
  int status;
  char* realname;
  realname = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
  std::string name{realname};
  free(realname);
  return name;
}

TEST(TypeList, GetSize)
{
  static_assert(GetSize<Types<>> == 0);
  static_assert(GetSize<Types<int>> == 1);
  static_assert(GetSize<Types<int, int>> == 2);
  static_assert(GetSize<Types<int, void>> == 2);
}

TEST(TypeList, GetType)
{
  EXPECT_SAME_TYPE((GetType<int, 0>), int);
  EXPECT_SAME_TYPE((GetType<Types<int, float, double>, 0>), int);
  EXPECT_SAME_TYPE((GetType<Types<int, float, double>, 1>), float);
  EXPECT_SAME_TYPE((GetType<Types<int, float, double>, 2>), double);
}

TEST(TypeList, Concat)
{
  EXPECT_SAME_TYPE(Concat<>, Types<>);
  EXPECT_SAME_TYPE((Concat<Types<long, void*, char*>>), (Types<long, void*, char*>));

  EXPECT_SAME_TYPE((Concat<Types<long, void*, char*>, Types<float, char, double>>),
                   (Types<long, void*, char*, float, char, double>));

  EXPECT_SAME_TYPE(
    (Concat<Types<long, void*, char*>, Types<float, char, double>, Types<int*, long*, unsigned>>),
    (Types<long, void*, char*, float, char, double, int*, long*, unsigned>));
}

TEST(TypeList, Flatten)
{
  EXPECT_SAME_TYPE(Flatten<Types<>>, Types<>);
  EXPECT_SAME_TYPE(Flatten<Types<int>>, Types<int>);
  EXPECT_SAME_TYPE((Flatten<Types<int, double>>), (Types<int, double>));
  EXPECT_SAME_TYPE((Flatten<Types<Types<int, double>, float>>), (Types<int, double, float>));
  EXPECT_SAME_TYPE((Flatten<Types<Types<int, Types<double>>, float>>), (Types<int, double, float>));
}

TEST(TypeList, CrossProduct)
{
  EXPECT_SAME_TYPE(CrossProduct<>, Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<>, Types<int>>), Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<>, Types<int, double>>), Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<>, Types<int, double>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<>, Types<>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<int, double>, Types<>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<int>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<int, double>, Types<>>), Types<>);
  EXPECT_SAME_TYPE((CrossProduct<Types<int>, Types<int>>), (Types<Types<int, int>>));
  EXPECT_SAME_TYPE((CrossProduct<Types<int, double>, Types<int>>),
                   (Types<Types<int, int>, Types<double, int>>));
  EXPECT_SAME_TYPE((CrossProduct<Types<int>, Types<double, char>>),
                   (Types<Types<int, double>, Types<int, char>>));
  EXPECT_SAME_TYPE(
    (CrossProduct<Types<int, double>, Types<short, char>>),
    (Types<Types<int, short>, Types<int, char>, Types<double, short>, Types<double, char>>));
  EXPECT_SAME_TYPE((CrossProduct<Types<int, double>, Types<int>, Types<float>>),
                   (Types<Types<int, int, float>, Types<double, int, float>>));

  EXPECT_SAME_TYPE((CrossProduct<Types<int, double>, Types<int>, Types<float, char>>),
                   (Types<Types<int, int, float>,
                          Types<int, int, char>,
                          Types<double, int, float>,
                          Types<double, int, char>>));
  EXPECT_SAME_TYPE((CrossProduct<Types<int, double>, Types<int, short>, Types<float, char>>),
                   (Types<Types<int, int, float>,
                          Types<int, int, char>,
                          Types<int, short, float>,
                          Types<int, short, char>,
                          Types<double, int, float>,
                          Types<double, int, char>,
                          Types<double, short, float>,
                          Types<double, short, char>>));
  EXPECT_SAME_TYPE((CrossProduct<Types<int, float>, int>),
                   (Types<Types<int, int>, Types<float, int>>));
  EXPECT_SAME_TYPE((CrossProduct<int, Types<int, float>>),
                   (Types<Types<int, int>, Types<int, float>>));
}

TEST(TypeList, AllSame)
{
  static_assert(AllSame::Call<Types<int, int>>::value);
  static_assert(AllSame::Call<Types<int, int>>::value);
  static_assert(!AllSame::Call<Types<bool, int>>::value);

  static_assert(AllSame::Call<int, int>::value);
  static_assert(!AllSame::Call<int, bool>::value);

  static_assert(AllSame::Call<int, int, int>::value);
  static_assert(!AllSame::Call<int, float, int>::value);
  static_assert(!AllSame::Call<int, int, float>::value);
}

TEST(TypeList, Exists)
{
  static_assert(Exists<int, Types<int, char, float>>);
  static_assert(!Exists<int, Types<double, char, float>>);
  static_assert(!Exists<int, Types<>>);
  static_assert(Exists<int, Types<double, char, float, int>>);
  static_assert(!Exists<int, Types<double>>);
  static_assert(Exists<int, Types<int>>);
}

TEST(TypeList, ContainedIn)
{
  static_assert(ContainedIn<Types<Types<int, char>>>::Call<Types<int, char>>::value);
  static_assert(!ContainedIn<Types<Types<int, char>>>::Call<Types<int, float>>::value);
  static_assert(!ContainedIn<Types<>>::Call<Types<int, float>>::value);
  static_assert(
    ContainedIn<Types<Types<int, float>, Types<char, char>>>::Call<Types<int, float>>::value);
  static_assert(
    !ContainedIn<Types<Types<int, float>, Types<char, char>>>::Call<Types<int, double>>::value);
  static_assert(ContainedIn<Types<Types<int, float>, Types<>>>::Call<Types<>>::value);
  static_assert(!ContainedIn<Types<Types<int, float>, Types<int>>>::Call<Types<>>::value);
}

TEST(TypeList, RemoveIf)
{
  EXPECT_SAME_TYPE((RemoveIf<AllSame, Types<>>), Types<>);

  EXPECT_SAME_TYPE((RemoveIf<AllSame, Types<Types<int, int, int>>>), Types<>);

  EXPECT_SAME_TYPE((RemoveIf<AllSame, Types<Types<int, float, int>>>),
                   (Types<Types<int, float, int>>));

  EXPECT_SAME_TYPE(
    (RemoveIf<AllSame,
              Types<Types<int, float, char>, Types<int, int, int>, Types<int, int, char>>>),
    (Types<Types<int, float, char>, Types<int, int, char>>));

  EXPECT_SAME_TYPE(
    (RemoveIf<AllSame,
              Types<Types<int, float, char>, Types<int, float, char>, Types<int, int, char>>>),
    (Types<Types<int, float, char>, Types<int, float, char>, Types<int, int, char>>));

  EXPECT_SAME_TYPE((RemoveIf<ContainedIn<Types<Types<int, char>, Types<float, int>>>,
                             Types<Types<char, char>, Types<float, int>, Types<int, int>>>),
                   (Types<Types<char, char>, Types<int, int>>));
}

TEST(TypeList, Transform)
{
  EXPECT_SAME_TYPE((Transform<Repeat<2>, Types<int, float>>),
                   (Types<Types<int, int>, Types<float, float>>));

  EXPECT_SAME_TYPE((Transform<Repeat<1>, Types<int, float>>), (Types<Types<int>, Types<float>>));
  EXPECT_SAME_TYPE((Transform<Repeat<0>, Types<int, float>>), (Types<Types<>, Types<>>));
  EXPECT_SAME_TYPE((Transform<Repeat<2>, Types<int>>), (Types<Types<int, int>>));
  EXPECT_SAME_TYPE((Transform<Repeat<1>, Types<int>>), Types<Types<int>>);
  EXPECT_SAME_TYPE((Transform<Repeat<0>, Types<>>), Types<>);
}

TEST(TypeList, Append)
{
  EXPECT_SAME_TYPE(Append<Types<>>, Types<>);
  EXPECT_SAME_TYPE((Append<Types<>, int>), Types<int>);
  EXPECT_SAME_TYPE(Append<Types<int>>, Types<int>);
  EXPECT_SAME_TYPE((Append<Types<int>, float>), (Types<int, float>));
  EXPECT_SAME_TYPE((Append<Types<int>, float, char>), (Types<int, float, char>));
}

TEST(TypeList, Remove)
{
  EXPECT_SAME_TYPE((Remove<Types<int, float, char>, 1>), (Types<int, char>));
  EXPECT_SAME_TYPE((Remove<Types<int, float, char>, 0, 2>), Types<float>);
  EXPECT_SAME_TYPE((Remove<Types<int, char>>), (Types<int, char>));
  EXPECT_SAME_TYPE((Remove<Types<int>, 0>), Types<>);
  EXPECT_SAME_TYPE((Remove<Types<int, char>, 0, 1>), Types<>);
  EXPECT_SAME_TYPE(Remove<Types<>>, Types<>);
}

TEST(TypeList, Unique)
{
  EXPECT_SAME_TYPE(Unique<Types<>>, Types<>);
  EXPECT_SAME_TYPE((Unique<Types<int, char, float>>), (Types<int, char, float>));
  EXPECT_SAME_TYPE((Unique<Types<int, int, float>>), (Types<int, float>));
  EXPECT_SAME_TYPE((Unique<Types<int, char, char, float>>), (Types<int, char, float>));
  EXPECT_SAME_TYPE((Unique<Types<int, char, float, float>>), (Types<int, char, float>));
  EXPECT_SAME_TYPE((Unique<Types<int, float, int, float>>), (Types<int, float>));
}
