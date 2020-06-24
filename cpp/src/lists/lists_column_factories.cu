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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>

namespace cudf {

/**
 * @copydoc cudf::make_lists_column
 *
 */
std::unique_ptr<column> make_lists_column(size_type num_rows,
                                          std::unique_ptr<column> offsets_column,
                                          std::unique_ptr<column> child_column,
                                          size_type null_count,
                                          rmm::device_buffer&& null_mask,
                                          cudaStream_t stream,
                                          rmm::mr::device_memory_resource* mr)
{
  if (null_count > 0) { CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable."); }
  CUDF_EXPECTS(num_rows == offsets_column->size() - 1,
               "Invalid offsets column size for lists column.");
  CUDF_EXPECTS(offsets_column->null_count() == 0, "Offsets column should not contain nulls");

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(child_column));
  return std::make_unique<column>(cudf::data_type{type_id::LIST},
                                  num_rows,
                                  rmm::device_buffer{0, stream, mr},
                                  null_mask,
                                  null_count,
                                  std::move(children));
}

}  // namespace cudf
