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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>

namespace {
// Returns true if the mask is true and valid (non-null) for index i
// This is the filter functor for apply_boolean_mask
template <bool has_nulls = true>
struct boolean_mask_filter {
  boolean_mask_filter(cudf::column_device_view const& boolean_mask) : boolean_mask{boolean_mask} {}

  __device__ inline bool operator()(cudf::size_type i)
  {
    if (true == has_nulls) {
      bool valid   = boolean_mask.is_valid(i);
      bool is_true = boolean_mask.data<bool>()[i];

      return is_true && valid;
    } else {
      return boolean_mask.data<bool>()[i];
    }
  }

 protected:
  cudf::column_device_view boolean_mask;
};

}  // namespace

namespace cudf {
namespace detail {
/*
 * Filters a table_view using a column_view of boolean values as a mask.
 *
 * calls copy_if() with the `boolean_mask_filter` functor.
 */
std::unique_ptr<table> apply_boolean_mask(table_view const& input,
                                          column_view const& boolean_mask,
                                          rmm::mr::device_memory_resource* mr,
                                          cudaStream_t stream)
{
  if (boolean_mask.is_empty()) { return empty_like(input); }

  CUDF_EXPECTS(boolean_mask.type().id() == type_id::BOOL8, "Mask must be Boolean type");
  // zero-size inputs are OK, but otherwise input size must match mask size
  CUDF_EXPECTS(input.num_rows() == 0 || input.num_rows() == boolean_mask.size(),
               "Column size mismatch");

  auto device_boolean_mask = cudf::column_device_view::create(boolean_mask, stream);

  if (boolean_mask.has_nulls()) {
    return detail::copy_if(input, boolean_mask_filter<true>{*device_boolean_mask}, mr, stream);
  } else {
    return detail::copy_if(input, boolean_mask_filter<false>{*device_boolean_mask}, mr, stream);
  }
}

}  // namespace detail

/*
 * Filters a table_view using a column_view of boolean values as a mask.
 */
std::unique_ptr<table> apply_boolean_mask(table_view const& input,
                                          column_view const& boolean_mask,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_boolean_mask(input, boolean_mask, mr);
}
}  // namespace cudf
