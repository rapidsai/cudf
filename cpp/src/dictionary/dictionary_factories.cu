/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace {
struct dispatch_create_indices {
  template <typename IndexType, std::enable_if_t<is_index_type<IndexType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& indices,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    CUDF_EXPECTS(std::is_unsigned<IndexType>(), "indices must be an unsigned type");
    column_view indices_view{
      indices.type(), indices.size(), indices.data<IndexType>(), nullptr, 0, indices.offset()};
    return std::make_unique<column>(indices_view, stream, mr);
  }
  template <typename IndexType, std::enable_if_t<!is_index_type<IndexType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::mr::device_memory_resource*,
                                     cudaStream_t)
  {
    CUDF_FAIL("indices must be an integer type.");
  }
};
}  // namespace

std::unique_ptr<column> make_dictionary_column(column_view const& keys_column,
                                               column_view const& indices_column,
                                               rmm::mr::device_memory_resource* mr,
                                               cudaStream_t stream)
{
  CUDF_EXPECTS(!keys_column.has_nulls(), "keys column must not have nulls");
  if (keys_column.size() == 0) return make_empty_column(data_type{type_id::DICTIONARY32});

  auto keys_copy = std::make_unique<column>(keys_column, stream, mr);
  auto indices_copy =
    type_dispatcher(indices_column.type(), dispatch_create_indices{}, indices_column, mr, stream);
  rmm::device_buffer null_mask{0, stream, mr};
  auto null_count = indices_column.null_count();
  if (null_count) null_mask = copy_bitmask(indices_column, stream, mr);

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(indices_copy));
  children.emplace_back(std::move(keys_copy));
  return std::make_unique<column>(data_type{type_id::DICTIONARY32},
                                  indices_column.size(),
                                  rmm::device_buffer{0, stream, mr},
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

std::unique_ptr<column> make_dictionary_column(std::unique_ptr<column> keys_column,
                                               std::unique_ptr<column> indices_column,
                                               rmm::device_buffer&& null_mask,
                                               size_type null_count)
{
  CUDF_EXPECTS(!keys_column->has_nulls(), "keys column must not have nulls");
  CUDF_EXPECTS(!indices_column->has_nulls(), "indices column must not have nulls");
  CUDF_EXPECTS(is_unsigned(indices_column->type()), "indices must be type unsigned integer");

  auto count = indices_column->size();
  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(indices_column));
  children.emplace_back(std::move(keys_column));
  return std::make_unique<column>(data_type{type_id::DICTIONARY32},
                                  count,
                                  rmm::device_buffer{},
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

}  // namespace cudf
