/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>
#include <vector>

using namespace cudf::test;

// =============================================================================
// ---- test data --------------------------------------------------------------

namespace {
namespace testdata {
// ----- most numerics

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, fixed_width_column_wrapper<T>>
ascending()
{
  return std::is_signed_v<T> ? fixed_width_column_wrapper<T>({std::numeric_limits<T>::lowest(),
                                                              T(-100),
                                                              T(-10),
                                                              T(-1),
                                                              T(0),
                                                              T(1),
                                                              T(10),
                                                              T(100),
                                                              std::numeric_limits<T>::max()})
                             : fixed_width_column_wrapper<T>({std::numeric_limits<T>::lowest(),
                                                              T(0),
                                                              T(1),
                                                              T(10),
                                                              T(100),
                                                              std::numeric_limits<T>::max()});
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, fixed_width_column_wrapper<T>>
descending()
{
  return std::is_signed_v<T> ? fixed_width_column_wrapper<T>({std::numeric_limits<T>::max(),
                                                              T(100),
                                                              T(10),
                                                              T(1),
                                                              T(0),
                                                              T(-1),
                                                              T(-10),
                                                              T(-100),
                                                              std::numeric_limits<T>::lowest()})
                             : fixed_width_column_wrapper<T>({std::numeric_limits<T>::max(),
                                                              T(100),
                                                              T(10),
                                                              T(1),
                                                              T(0),
                                                              std::numeric_limits<T>::lowest()});
}

template <typename T>
auto empty()
{
  return fixed_width_column_wrapper<T>();
}

template <typename T>
auto nulls_after()
{
  return fixed_width_column_wrapper<T, int32_t>({0, 0}, {1, 0});
}

template <typename T>
auto nulls_before()
{
  return fixed_width_column_wrapper<T, int32_t>({0, 0}, {0, 1});
}

// ----- bool

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, fixed_width_column_wrapper<bool>> ascending()
{
  return fixed_width_column_wrapper<bool>({false, false, true, true});
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, fixed_width_column_wrapper<bool>> descending()
{
  return fixed_width_column_wrapper<bool>({true, true, false, false});
}

// ----- chrono types

template <typename T>
std::enable_if_t<cudf::is_chrono<T>(), fixed_width_column_wrapper<T>> ascending()
{
  return fixed_width_column_wrapper<T>({T::min(), T::max()});
}

template <typename T>
std::enable_if_t<cudf::is_chrono<T>(), fixed_width_column_wrapper<T>> descending()
{
  return fixed_width_column_wrapper<T>({T::max(), T::min()});
}

// ----- string_view

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, strings_column_wrapper> ascending()
{
  return strings_column_wrapper({"A", "B"});
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, strings_column_wrapper> descending()
{
  return strings_column_wrapper({"B", "A"});
}

template <>
auto empty<cudf::string_view>()
{
  return strings_column_wrapper();
}

template <>
auto nulls_after<cudf::string_view>()
{
  return strings_column_wrapper({"identical", "identical"}, {1, 0});
}

template <>
auto nulls_before<cudf::string_view>()
{
  return strings_column_wrapper({"identical", "identical"}, {0, 1});
}

// ----- struct_view {"nestedInt" : {"Int" : 0 }, "float" : 1}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::struct_view>, structs_column_wrapper> ascending()
{
  using T1           = int32_t;
  auto int_col       = fixed_width_column_wrapper<int32_t>({std::numeric_limits<T1>::lowest(),
                                                      T1(-100),
                                                      T1(-10),
                                                      T1(-10),
                                                      T1(0),
                                                      T1(10),
                                                      T1(10),
                                                      T1(100),
                                                      std::numeric_limits<T1>::max()});
  auto nestedInt_col = structs_column_wrapper{{int_col}};
  auto float_col     = ascending<float>();
  return structs_column_wrapper{{nestedInt_col, float_col}};
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::struct_view>, structs_column_wrapper> descending()
{
  using T1           = int32_t;
  auto int_col       = fixed_width_column_wrapper<int32_t>({std::numeric_limits<T1>::max(),
                                                      T1(100),
                                                      T1(10),
                                                      T1(10),
                                                      T1(0),
                                                      T1(-10),
                                                      T1(-10),
                                                      T1(-100),
                                                      std::numeric_limits<T1>::lowest()});
  auto nestedInt_col = structs_column_wrapper{{int_col}};
  auto float_col     = descending<float>();
  return structs_column_wrapper{{nestedInt_col, float_col}};
}

template <>
auto empty<cudf::struct_view>()
{
  auto int_col = fixed_width_column_wrapper<int32_t>();
  auto col1    = structs_column_wrapper{{int_col}};
  auto col2    = fixed_width_column_wrapper<float>();
  return structs_column_wrapper{{col1, col2}};
}

template <>
auto nulls_after<cudf::struct_view>()
{
  auto int_col = fixed_width_column_wrapper<int32_t>({1, 1});
  auto col1    = structs_column_wrapper{{int_col}};
  auto col2    = fixed_width_column_wrapper<float>({1, 1});
  return structs_column_wrapper{{col1, col2}, {1, 0}};
}

template <>
auto nulls_before<cudf::struct_view>()
{
  auto int_col = fixed_width_column_wrapper<int32_t>({1, 1});
  auto col1    = structs_column_wrapper{{int_col}};
  auto col2    = fixed_width_column_wrapper<float>({1, 1});
  return structs_column_wrapper{{col1, col2}, {0, 1}};
}

}  // namespace testdata
}  // anonymous namespace

// =============================================================================
// ---- tests ------------------------------------------------------------------

template <typename T>
struct IsSortedTest : public BaseFixture {
};

using SupportedTypes = Concat<ComparableTypes, cudf::test::Types<cudf::struct_view>>;
TYPED_TEST_SUITE(IsSortedTest, SupportedTypes);

TYPED_TEST(IsSortedTest, NoColumns)
{
  cudf::table_view in{std::vector<cudf::table_view>{}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, NoRows)
{
  using T = TypeParam;

  if (std::is_same_v<T, cudf::string_view>) {
    // strings_column_wrapper does not yet support empty columns.
    return;
  } else {
    auto col1 = testdata::empty<T>();
    auto col2 = testdata::empty<T>();

    cudf::table_view in{{col1, col2}};
    std::vector<cudf::order> order{cudf::order::ASCENDING, cudf::order::DESCENDING};
    std::vector<cudf::null_order> null_precedence{};

    auto actual = cudf::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
  }
}

TYPED_TEST(IsSortedTest, Ascending)
{
  using T = TypeParam;

  auto col1 = testdata::ascending<T>();
  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, AscendingFalse)
{
  using T = TypeParam;

  auto col1 = testdata::descending<T>();
  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, {});

  EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedTest, Descending)
{
  using T = TypeParam;

  auto col1 = testdata::descending<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, DescendingFalse)
{
  using T = TypeParam;

  auto col1 = testdata::ascending<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedTest, NullsAfter)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_after<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, NullsAfterFalse)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_before<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedTest, NullsBefore)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_before<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, NullsBeforeFalse)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_after<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedTest, OrderArgsTooFew)
{
  using T = TypeParam;

  auto col1 = testdata::ascending<T>();
  auto col2 = testdata::ascending<T>();

  cudf::table_view in{{col1, col2}};
  std::vector<cudf::order> order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  EXPECT_THROW(cudf::is_sorted(in, order, null_precedence), cudf::logic_error);
}

TYPED_TEST(IsSortedTest, OrderArgsTooMany)
{
  using T = TypeParam;

  auto col1 = testdata::ascending<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  EXPECT_THROW(cudf::is_sorted(in, order, null_precedence), cudf::logic_error);
}

TYPED_TEST(IsSortedTest, NullOrderArgsTooFew)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_before<T>();
  auto col2 = testdata::nulls_before<T>();

  cudf::table_view in{{col1, col2}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE};

  EXPECT_THROW(cudf::is_sorted(in, order, null_precedence), cudf::logic_error);
}

TYPED_TEST(IsSortedTest, NullOrderArgsTooMany)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_before<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE, cudf::null_order::BEFORE};

  EXPECT_THROW(cudf::is_sorted(in, order, null_precedence), cudf::logic_error);
}

template <typename T>
struct IsSortedFixedWidthOnly : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(IsSortedFixedWidthOnly, cudf::test::FixedWidthTypes);

CUDF_TEST_PROGRAM_MAIN()
