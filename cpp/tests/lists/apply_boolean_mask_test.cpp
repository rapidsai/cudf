/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/stream_compaction.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>
#include <tests/strings/utilities.h>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <vector>

namespace cudf::test {

using cudf::lists_column_view;
using cudf::lists::apply_boolean_mask;

using lists_t  = lists_column_wrapper<int32_t>;
using filter_t = lists_column_wrapper<bool, int32_t>;

struct ApplyBooleanMaskTest : public BaseFixture {
};

template <typename T>
struct ApplyBooleanMaskTypedTest : ApplyBooleanMaskTest {
};

using TestTypes = cudf::test::NumericTypes;

TEST_F(ApplyBooleanMaskTest, StraightLine)
{
  auto input  = lists_t{{0, 1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {0, 1}, {2, 3, 4, 5}, {6, 7}}.release();
  auto filter = filter_t{{1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}};

  {
    // Unsliced.
    auto filtered = apply_boolean_mask(lists_column_view{*input}, lists_column_view{filter});
    auto expected = lists_t{{0, 2}, {4}, {6, 8}, {0}, {2, 4}, {6}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
  {
    // Sliced input: Remove the first row.
    auto sliced = cudf::slice(*input, {1, input->size()}).front();
    //         == lists_t {{4, 5}, {6, 7, 8, 9}, {0, 1}, {2, 3, 4, 5}, {6, 7}};
    auto filter   = filter_t{{0, 1}, {0, 1, 0, 1}, {1, 1}, {0, 1, 0, 1}, {0, 0}};
    auto filtered = apply_boolean_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists_t{{5}, {7, 9}, {0, 1}, {3, 5}, {}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
}

/*
TEST_F(SegmentedFilterTest, TheBasic)
{
  std::cout << "CALEB: SegmentedFilterTest.TheBasic" << std::endl;
  auto filteree = lists{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 0, 1}, {2, 3, 4}, {5, 6, 7}};
  auto filterer =
    filter{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}, {0, 1, 0}, {1, 0, 1}, {0, 1, 0}}.release();
  // auto filterer   = filter{{1,0,1}, {}, {1,0,1}, {0,1,0}, {1,0,1}, {0,1,0}}.release();
  auto bools_cv = lists_column_view(*filterer);
  std::cout << "Filteree: " << std::endl;
  print(filteree);
  std::cout << "Filterer: " << std::endl;
  print(*filterer);

  auto output_offsets = cudf::reduction::segmented_sum(
    bools_cv.child(), bools_cv.offsets(), data_type{type_id::INT32}, null_policy::EXCLUDE);
  std::cout << "Output offsets:\n";
  print(*output_offsets);

  auto filtered_child = cudf::detail::apply_boolean_mask(
    cudf::table_view{{lists_column_view{filteree}.child()}}, bools_cv.child());
  std::cout << "Filtered child:\n";
  print(filtered_child->view().column(0));
}

TEST_F(SegmentedFilterTest, TheBasicForNestedLists)
{
  std::cout << "CALEB: SegmentedFilterTest.TheBasicForNestedLists" << std::endl;
  auto filteree =
    lists{{{0, 1}, {2}}, {{3}, {4, 5}}, {{6, 7}, {8}}, {{9, 0, 1}}, {{2, 3, 4}}, {{5, 6, 7}}};
  auto filterer = filter{{1, 0}, {0, 1}, {1, 0}, {0}, {1}, {0}}.release();
  // auto filterer   = filter{{1,0,1}, {}, {1,0,1}, {0,1,0}, {1,0,1}, {0,1,0}}.release();
  auto bools_cv = lists_column_view(*filterer);
  std::cout << "Filteree: " << std::endl;
  print(filteree);
  std::cout << "Filterer: " << std::endl;
  print(*filterer);

  auto output_offsets = cudf::reduction::segmented_sum(
    bools_cv.child(), bools_cv.offsets(), data_type{type_id::INT32}, null_policy::EXCLUDE);
  std::cout << "Output offsets:\n";
  print(*output_offsets);

  auto filtered_child = cudf::detail::apply_boolean_mask(
    cudf::table_view{{lists_column_view{filteree}.child()}}, bools_cv.child());
  std::cout << "Filtered child:\n";
  print(filtered_child->view().column(0));
}
*/

}  // namespace cudf::test
