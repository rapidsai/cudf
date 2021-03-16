/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace test {

#define COL_K cudf::test::fixed_width_column_wrapper<int32_t, int32_t>
#define COL_V cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>
#define LCL_V cudf::test::lists_column_wrapper<TypeParam, int32_t>
#define DCL_V cudf::test::dictionary_column_wrapper<TypeParam, int32_t>
#define VALIDITY std::initializer_list<bool>
#define COLLECT_SET cudf::make_collect_list_aggregation()

template <typename V>
struct CollectSetTest : public cudf::test::BaseFixture {
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(CollectSetTest, FixedWidthTypesNotBool);

TYPED_TEST(CollectSetTest, ExceptionCases)
{
  std::vector<groupby::aggregation_request> agg_requests(1);
  agg_requests[0].values = COL_V{{1, 2, 3, 4, 5, 6}, {true, false, true, false, true, false}};
  agg_requests[0].aggregations.push_back(cudf::make_collect_list_aggregation(null_policy::EXCLUDE));

  // groupby cannot exclude nulls
  groupby::groupby gby{table_view{{COL_K{1, 1, 2, 2, 3, 3}}}};
  EXPECT_THROW(gby.aggregate(agg_requests), cudf::logic_error);
}

// TODO: Fix those cases to handle empty and simple input
TYPED_TEST(CollectSetTest, DISABLED_TrivialCases)
{
  // Empty input
  test_single_agg(COL_K{}, COL_V{}, COL_K{}, COL_V{}, COLLECT_SET);

  // Single key input
  {
    COL_K keys{1};
    COL_K keys_expected{1};
    COL_V vals{100};
    COL_V vals_expected{{100}};
    test_single_agg(keys, vals, keys_expected, vals_expected, COLLECT_SET);
  }

  // Simple input
  {
    COL_K keys{1, 2};
    COL_K keys_expected{1, 2};
    COL_V vals{100, 200};
    LCL_V vals_expected{{100}, {200}};
    test_single_agg(keys, vals, keys_expected, vals_expected, COLLECT_SET);
  }
}

TYPED_TEST(CollectSetTest, TypicalCases)
{
  // Hard-coded case
  {
    COL_K keys{1, 1, 2, 2, 3, 3};
    COL_K keys_expected{1, 2, 3};
    LCL_V vals{{1, 2}, {3, 4}, {5, 6, 7}, {}, {9, 10}, {11}};
    LCL_V vals_expected{{{1, 2}, {3, 4}}, {{5, 6, 7}, {}}, {{9, 10}, {11}}};
    test_single_agg(keys, vals, keys_expected, vals_expected, COLLECT_SET);
  }

  // Procedurally generated test
  {
    COL_K keys{1, 1, 2, 2, 3, 3};
    COL_K keys_expected{1, 2, 3};
    LCL_V vals{{1, 2}, {3, 4}, {5, 6, 7}, {}, {9, 10}, {11}};
    LCL_V vals_expected{{{1, 2}, {3, 4}}, {{5, 6, 7}, {}}, {{9, 10}, {11}}};
    test_single_agg(keys, vals, keys_expected, vals_expected, COLLECT_SET);
  }
}

TYPED_TEST(CollectSetTest, CollectWithNulls)
{
  // Hard-coded case
  {
    COL_K keys{1, 1, 2, 2, 3, 3};
    COL_K keys_expected{1, 2, 3};
    COL_V vals{{1, 2, 3, 4, 5, 6}, {true, false, true, false, true, false}};
    LCL_V vals_expected{{{1, 2}, VALIDITY{true, false}},
                        {{3, 4}, VALIDITY{true, false}},
                        {{5, 6}, VALIDITY{true, false}}};
    test_single_agg(keys, vals, keys_expected, vals_expected, COLLECT_SET);
  }

  // Procedurally generated test
  {
    //
  }
}

}  // namespace test
}  // namespace cudf

CUDF_TEST_PROGRAM_MAIN()
