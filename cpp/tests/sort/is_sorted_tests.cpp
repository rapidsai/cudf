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

#include <cudf/column/column_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <vector>

using namespace cudf::test;

// =============================================================================
// ---- test data --------------------------------------------------------------

namespace {
namespace testdata {
// ----- most numerics

template <typename T>
auto ascending()
{
  return std::is_signed<T>::value ? fixed_width_column_wrapper<T>({std::numeric_limits<T>::lowest(),
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
auto descending()
{
  return std::is_signed<T>::value
           ? fixed_width_column_wrapper<T>({std::numeric_limits<T>::max(),
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
  return fixed_width_column_wrapper<T>({0, 0}, {1, 0});
}

template <typename T>
auto nulls_before()
{
  return fixed_width_column_wrapper<T>({0, 0}, {0, 1});
}

// ----- bool

template <>
auto ascending<bool>()
{
  return fixed_width_column_wrapper<bool>({false, false, true, true});
}

template <>
auto descending<bool>()
{
  return fixed_width_column_wrapper<bool>({true, true, false, false});
}

// ----- timestamp

template <typename T>
fixed_width_column_wrapper<T> ascending_timestamp()
{
  return fixed_width_column_wrapper<T>(
    {T::min().time_since_epoch().count(), T::max().time_since_epoch().count()});
}

template <typename T>
fixed_width_column_wrapper<T> descending_timestamp()
{
  return fixed_width_column_wrapper<T>(
    {T::max().time_since_epoch().count(), T::min().time_since_epoch().count()});
}

template <>
auto ascending<cudf::timestamp_D>()
{
  return ascending_timestamp<cudf::timestamp_D>();
}
template <>
auto ascending<cudf::timestamp_s>()
{
  return ascending_timestamp<cudf::timestamp_s>();
}
template <>
auto ascending<cudf::timestamp_ms>()
{
  return ascending_timestamp<cudf::timestamp_ms>();
}
template <>
auto ascending<cudf::timestamp_us>()
{
  return ascending_timestamp<cudf::timestamp_us>();
}
template <>
auto ascending<cudf::timestamp_ns>()
{
  return ascending_timestamp<cudf::timestamp_ns>();
}

template <>
auto descending<cudf::timestamp_D>()
{
  return descending_timestamp<cudf::timestamp_D>();
}
template <>
auto descending<cudf::timestamp_s>()
{
  return descending_timestamp<cudf::timestamp_s>();
}
template <>
auto descending<cudf::timestamp_ms>()
{
  return descending_timestamp<cudf::timestamp_ms>();
}
template <>
auto descending<cudf::timestamp_us>()
{
  return descending_timestamp<cudf::timestamp_us>();
}
template <>
auto descending<cudf::timestamp_ns>()
{
  return descending_timestamp<cudf::timestamp_ns>();
}

// ----- string_view

template <>
auto ascending<cudf::string_view>()
{
  return strings_column_wrapper({"A", "B"});
}

template <>
auto descending<cudf::string_view>()
{
  return strings_column_wrapper({"B", "A"});
}

template <>
auto empty<cudf::string_view>()
{
  return strings_column_wrapper({});
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

}  // namespace testdata
}  // anonymous namespace

// =============================================================================
// ---- tests ------------------------------------------------------------------

template <typename T>
struct IsSortedTest : public BaseFixture {
};

TYPED_TEST_CASE(IsSortedTest, ComparableTypes);

TYPED_TEST(IsSortedTest, NoColumns)
{
  using T = TypeParam;

  cudf::table_view in{std::vector<cudf::table_view>{}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, NoRows)
{
  using T = TypeParam;

  if (std::is_same<T, cudf::string_view>::value) {
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

TYPED_TEST_CASE(IsSortedFixedWidthOnly, cudf::test::FixedWidthTypes);

CUDF_TEST_PROGRAM_MAIN()
