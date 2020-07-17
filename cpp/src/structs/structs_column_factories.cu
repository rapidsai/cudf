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
#include <cudf/detail/null_mask.hpp>
#include <algorithm>

namespace cudf
{
  namespace 
  {
    // Helper function to superimpose validity of parent struct
    // over all member fields (i.e. child columns).
    void superimpose_validity(
      rmm::device_buffer const& parent_null_mask, 
      size_type parent_null_count,
      std::vector<std::unique_ptr<column>>& children,
      cudaStream_t stream,
      rmm::mr::device_memory_resource* mr
    )
    {
      if (parent_null_mask.is_empty()) {
        // Struct is not nullable. Children do not need adjustment.
        // Bail.
        return;
      }

      std::for_each(
        children.begin(),
        children.end(),
        [&](std::unique_ptr<column>& p_child)
        {
          if (!p_child->nullable())
          {
            p_child->set_null_mask(std::move(rmm::device_buffer{parent_null_mask, stream, mr})); 
            p_child->set_null_count(parent_null_count);
          }
          else {

            auto data_type{p_child->type()};
            auto num_rows{p_child->size()};

            // All this to reset the null mask. :/
            cudf::column::contents contents{p_child->release()};
            std::vector<bitmask_type const*> masks {
              reinterpret_cast<bitmask_type const*>(parent_null_mask.data()), 
              reinterpret_cast<bitmask_type const*>(contents.null_mask->data())};
            
            rmm::device_buffer new_child_mask = cudf::detail::bitmask_and(masks, {0, 0}, num_rows, stream, mr);

            // Recurse for struct members.
            // Push down recomputed child mask to child columns of the current child.
            if (data_type.id() == cudf::type_id::STRUCT)
            {
              superimpose_validity(new_child_mask, UNKNOWN_NULL_COUNT, contents.children, stream, mr);
            }

            // Reconstitute the column.
            p_child.reset(
              new column(
                data_type,
                num_rows,
                std::move(*contents.data),
                std::move(new_child_mask),
                UNKNOWN_NULL_COUNT,
                std::move(contents.children)
              )
            );
          }
        }
      );
    }
  }

  /// Column factory that adopts child columns.
  std::unique_ptr<cudf::column> make_structs_column(
    size_type num_rows,
    std::vector<std::unique_ptr<column>>&& child_columns,
    size_type null_count,
    rmm::device_buffer&& null_mask,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
  {
      if (null_count > 0)
      {
        CUDF_EXPECTS(!null_mask.is_empty(), "Column with nulls must be nullable.");
      }

      CUDF_EXPECTS(
        std::all_of(child_columns.begin(), 
                    child_columns.end(), 
                    [&](auto const& i){ return num_rows == i->size(); }), 
        "Child columns must have the same number of rows as the Struct column.");

      superimpose_validity(null_mask, null_count, child_columns, stream, mr);

      return std::make_unique<column>(
        cudf::data_type{type_id::STRUCT},
        num_rows,
        rmm::device_buffer{0, stream, mr}, // Empty data buffer. Structs hold no data.
        null_mask,
        null_count,
        std::move(child_columns)
      );
  }

} // namespace cudf;
