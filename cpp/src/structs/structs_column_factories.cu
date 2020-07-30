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
#include <memory>
#include "cudf/types.hpp"
#include "thrust/iterator/counting_iterator.h"

namespace cudf
{
  namespace 
  {
    // Helper function to superimpose validity of parent struct
    // over the specified member (child) column.
    void superimpose_parent_nullmask(
      rmm::device_buffer const& parent_null_mask, 
      size_type parent_null_count,
      column& child,
      cudaStream_t stream,
      rmm::mr::device_memory_resource* mr
    )
    {
      if (!child.nullable())
      {
        child.set_null_mask(std::move(rmm::device_buffer{parent_null_mask, stream, mr})); 
        child.set_null_count(parent_null_count);
      }
      else {

        auto data_type{child.type()};
        auto num_rows{child.size()};

        auto new_child_mask =
          cudf::detail::bitmask_and(
            {
              reinterpret_cast<bitmask_type const*>(parent_null_mask.data()),
              reinterpret_cast<bitmask_type const*>(child.mutable_view().null_mask())
            },
            {0, 0},
            child.size(),
            stream,
            mr
          );

        if (child.type().id() == cudf::type_id::STRUCT)
        {
          std::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(child.num_children()),
            [&new_child_mask, &child, stream, mr](auto i) 
            {
              superimpose_parent_nullmask(new_child_mask, UNKNOWN_NULL_COUNT, child.child(i), stream, mr);
            }
          );
        }

        child.set_null_mask(std::move(new_child_mask));
        child.set_null_count(UNKNOWN_NULL_COUNT);
      }
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

      if (!null_mask.is_empty())
      {
        std::for_each(
          child_columns.begin(),
          child_columns.end(),
          [& null_mask,
             null_count,
             stream,
             mr
          ](auto & p_child) {
            superimpose_parent_nullmask(null_mask, null_count, *p_child, stream, mr);
          }
        );
      }

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
