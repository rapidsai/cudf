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

#include<cudf/cudf.h>
#include<cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/copy_if.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <algorithm>

namespace {

// Returns true if the mask is true and valid (non-null) for index i
// This is the filter functor for apply_boolean_mask
// Note we use a functor here so we can cast to a bitmask_t __restrict__
// pointer on the host side, which we can't do with a lambda.
template <typename T, bool has_nulls>
struct boolean_mask_filter
{
  boolean_mask_filter(cudf::column_device_view const& boolean_mask) :
    boolean_mask{boolean_mask}
    {}

  __device__ inline
  bool operator()(cudf::size_type i)
  {
    bool valid = !has_nulls || boolean_mask.is_valid(i);
    bool is_true = (cudf::experimental::true_v == boolean_mask.data<T>()[i]);
    
    return is_true && valid;
  }

protected:
  cudf::column_device_view boolean_mask;
};

}  // namespace

namespace cudf {
namespace experimental {
namespace detail {

/*
 * Filters a table using a column of boolean values as a mask.
 *
 * calls copy_if() with the `boolean_mask_filter` functor.
 */
std::unique_ptr<experimental::table> 
    apply_boolean_mask(table_view const& input,
                       column_view const& boolean_mask,
                       rmm::mr::device_memory_resource *mr,
                       cudaStream_t stream) {

  if (boolean_mask.size() == 0) {
      std::vector<std::unique_ptr<column>> out_columns(input.num_columns());
      std::transform(input.begin(), input.end(), out_columns.begin(),
                [&stream] (auto col_view){
                return detail::empty_like(col_view, stream);
                });

      return std::make_unique<experimental::table>(std::move(out_columns));
  }

  CUDF_EXPECTS(boolean_mask.type().id() == BOOL8, "Mask must be Boolean type");
  // zero-size inputs are OK, but otherwise input size must match mask size
  CUDF_EXPECTS(input.num_rows() == 0 || input.num_rows() == boolean_mask.size(),
               "Column size mismatch");

  auto device_boolean_mask = cudf::column_device_view::create(boolean_mask, stream);
  
  if(boolean_mask.nullable()){
    return detail::copy_if(input, 
                    boolean_mask_filter<cudf::experimental::bool8, true> {
                                                *device_boolean_mask
                                             });
  } else {
    return detail::copy_if(input, 
                    boolean_mask_filter<cudf::experimental::bool8, false> {
                                                *device_boolean_mask
                                             });
  }
}

} // namespace detail

std::unique_ptr<experimental::table>
    apply_boolean_mask(table_view const& input,
                       column_view const& boolean_mask,
                       rmm::mr::device_memory_resource *mr) {
    return detail::apply_boolean_mask(input, boolean_mask, mr);
}
} // namespace experimental
}  // namespace cudf


