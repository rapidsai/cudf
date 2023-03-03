/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/structs/utilities.hpp>
#include <cudf/sorting.hpp>

using int32s_lists = cudf::test::lists_column_wrapper<int32_t>;
using int32s_col   = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs_col  = cudf::test::structs_column_wrapper;

namespace {
std::vector<cudf::column_view> flatten_lists_of_structs(cudf::column_view const& input)
{
  if (input.type().id() != cudf::type_id::LIST ||
      input.child(cudf::lists_column_view::child_column_index).type().id() !=
        cudf::type_id::STRUCT) {
    return {input};
  }

  auto const offsets  = input.child(cudf::lists_column_view::offsets_column_index);
  auto const children = input.child(cudf::lists_column_view::child_column_index);
  std::vector<cudf::column_view> output;

  for (auto it = children.child_begin(); it != children.child_end(); ++it) {
    auto const new_column = cudf::column_view{cudf::data_type{cudf::type_id::LIST},
                                              input.size(),
                                              nullptr,
                                              input.null_mask(),
                                              input.null_count(),
                                              input.offset(),
                                              {offsets, *it}};
    // The new column may still be lists of structs, thus we recursively call this:
    auto const flattened_new_column = flatten_lists_of_structs(new_column);
    output.insert(output.end(), flattened_new_column.begin(), flattened_new_column.end());
  }

  return output;
}
}  // namespace

struct structs_test : public cudf::test::BaseFixture {
};

TEST_F(structs_test, StructsHaveLists)
{
  auto const input = [] {
    auto child1 = int32s_lists{{1, 1}, {2, 2}, {3, 3}, {1, 1}, {2, 2}};
    auto child2 = int32s_col{1, 2, 3, 1, 2};
    return structs_col{{child1, child2}};
  }();

  printf("line %d\n", __LINE__);
  cudf::test::print(input);

  if (1) {
    auto const order = cudf::sorted_order(cudf::table_view{{input}});

    printf("line %d\n", __LINE__);
    cudf::test::print(*order);

    auto const sorted = cudf::sort(cudf::table_view{{input}});

    printf("line %d\n", __LINE__);
    cudf::test::print(sorted->get_column(0).view());
  }

  {
    auto const flattened =
      cudf::structs::detail::flatten_nested_columns(cudf::table_view{{input}}, {}, {});
    auto const order = cudf::sorted_order(flattened.flattened_columns());

    printf("line %d\n", __LINE__);
    cudf::test::print(*order);

    auto const sorted = cudf::sort(cudf::table_view{{flattened.flattened_columns()}});

    printf("line %d\n", __LINE__);
    cudf::test::print(sorted->get_column(0).view());
  }
}

TEST_F(structs_test, ListsHaveStructs)
{
  auto const input = [] {
    auto const get_structs = [] {
      auto child1 = int32s_col{1, 2, 3, 0, 1, 2, 4, 5, 0, 1};
      auto child2 = int32s_col{11, 12, 13, 10, 11, 12, 14, 15, 10, 11};
      return structs_col{{child1, child2}};
    };

    return cudf::make_lists_column(
      4, int32s_col{0, 3, 6, 8, 10}.release(), get_structs().release(), 0, {});
  }();

  printf("line %d\n", __LINE__);
  cudf::test::print(*input);

  if (1) {
    auto const order = cudf::sorted_order(cudf::table_view{{*input}});

    printf("line %d\n", __LINE__);
    cudf::test::print(*order);

    auto const sorted = cudf::sort(cudf::table_view{{*input}});

    printf("line %d\n", __LINE__);
    cudf::test::print(sorted->get_column(0).view());
  }

  {
    auto const flattened = flatten_lists_of_structs(*input);

    auto const order = cudf::sorted_order(cudf::table_view{flattened});

    printf("line %d\n", __LINE__);
    cudf::test::print(*order);

    auto const sorted = cudf::sort(cudf::table_view{flattened});

    printf("line %d\n", __LINE__);
    cudf::test::print(sorted->get_column(0).view());
  }
}
