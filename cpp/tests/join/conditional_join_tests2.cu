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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/interop.hpp>
#include <cudf/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

struct ConditionalJoinTest2 : public cudf::test::BaseFixture {};

TEST_F(ConditionalJoinTest2, OutOfBoundJoinIndicesResult)
{
  auto make_table = [](int32_t size) -> std::unique_ptr<cudf::table> {
    auto sequence_column = cudf::sequence(size, cudf::numeric_scalar<int32_t>(0));

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(sequence_column));
    return std::make_unique<cudf::table>(std::move(columns));
  };

  auto split_into_int32_chunks = [](int64_t value) {
    std::vector<int32_t> result;
    auto chunk_size = static_cast<int64_t>(std::numeric_limits<int32_t>::max() - 1);
    while (value > static_cast<int64_t>(0)) {
      int64_t chunk = std::min(chunk_size, static_cast<int64_t>(value));
      result.push_back(static_cast<int64_t>(chunk));
      value -= chunk;
    }
    return result;
  };

  try {
    // Need a size that will produce more than std::numeric_limits<int32_t>::max() rows.
    // We use something slightly larger than the square root of that value, and return
    // true for every join predicate.
    auto const table_size = 50000;
    auto left_table       = make_table(table_size);
    auto right_table      = make_table(table_size);

    auto left_view  = left_table->view();
    auto right_view = right_table->view();

    std::cerr << "Left size: " << left_view.num_rows() << ", Right size: " << right_view.num_rows()
              << "\n";

    std::vector<int> join_column_indices = {0};
    cudf::table_view left_join_view      = left_view.select(join_column_indices);
    cudf::table_view right_join_view     = right_view.select(join_column_indices);

    auto true_scalar  = cudf::numeric_scalar<bool>(true);
    auto true_literal = cudf::ast::literal(true_scalar);
    auto join_size =
      cudf::conditional_inner_join_size(left_join_view, right_join_view, true_literal);
    std::cerr << "Join size: " << join_size << "\n";
    auto [left_indices, right_indices] =
      cudf::conditional_inner_join(left_join_view, right_join_view, true_literal);

    auto left_host = cudf::detail::make_std_vector_sync(*left_indices, cudf::get_default_stream());
    auto right_host =
      cudf::detail::make_std_vector_sync(*right_indices, cudf::get_default_stream());

    std::cerr << "Left indices: " << left_indices->size()
              << ", Right indices: " << right_indices->size() << "\n";
    ASSERT_TRUE(left_indices->size() > std::numeric_limits<int32_t>::max());
    ASSERT_TRUE(right_indices->size() > std::numeric_limits<int32_t>::max());

    auto chunks        = split_into_int32_chunks(left_indices->size());
    std::size_t offset = 0;
    for (auto chunk_size : chunks) {
      auto left_indices_span =
        cudf::device_span<cudf::size_type const>(left_indices->data() + offset, chunk_size);
      // auto right_indices_span = cudf::device_span<cudf::size_type const>(
      //     right_indices->data() + offset, chunk_size);

      cudf::column_view left_column{left_indices_span};
      std::cerr << ">> : cudf::minmax"
                << "\n";
      auto [min_val, max_val] = cudf::minmax(left_column);
      std::cerr << "<< : cudf::minmax"
                << "\n";

      cudf::column_metadata const metadata{""};
      auto arrow_min = cudf::to_arrow(*min_val, metadata);
      auto arrow_max = cudf::to_arrow(*max_val, metadata);

      std::cerr << "Min value: " << arrow_min->ToString()
                << ", Max value: " << arrow_max->ToString() << "\n";
      offset += chunk_size;

      auto left_joined = cudf::gather(left_view, cudf::column_view{left_indices_span});
    }

  } catch (const std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << "\n";
  }
}
