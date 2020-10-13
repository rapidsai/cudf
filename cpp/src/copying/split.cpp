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

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>

namespace cudf {
namespace {
template <typename T>
std::vector<T> split(T const& input, size_type column_size, std::vector<size_type> const& splits)
{
  if (splits.empty() or column_size == 0) { return std::vector<T>{input}; }
  CUDF_EXPECTS(splits.back() <= column_size, "splits can't exceed size of input columns");

  // If the size is not zero, the split will always start at `0`
  std::vector<size_type> indices{0};
  std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
    indices.push_back(split);  // This for end
    indices.push_back(split);  // This for the start
  });

  indices.push_back(column_size);  // This to include rest of the elements

  return cudf::slice(input, indices);
}
};  // anonymous namespace

std::vector<cudf::column_view> split(cudf::column_view const& input,
                                     std::vector<size_type> const& splits)
{
  CUDF_FUNC_RANGE();
  return split(input, input.size(), splits);
}

std::vector<cudf::table_view> split(cudf::table_view const& input,
                                    std::vector<size_type> const& splits)
{
  CUDF_FUNC_RANGE();
  std::vector<table_view> result{};
  if (input.num_columns() == 0) { return result; }
  return split(input, input.column(0).size(), splits);
}

}  // namespace cudf
