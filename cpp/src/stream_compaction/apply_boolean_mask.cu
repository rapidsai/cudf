/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

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
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  if (boolean_mask.is_empty()) { return empty_like(input); }

  CUDF_EXPECTS(boolean_mask.type().id() == type_id::BOOL8, "Mask must be Boolean type");
  // zero-size inputs are OK, but otherwise input size must match mask size
  CUDF_EXPECTS(input.num_rows() == 0 || input.num_rows() == boolean_mask.size(),
               "Column size mismatch");

  auto device_boolean_mask = cudf::column_device_view::create(boolean_mask, stream);

  if (boolean_mask.has_nulls()) {
    return detail::copy_if(input, boolean_mask_filter<true>{*device_boolean_mask}, stream, mr);
  } else {
    return detail::copy_if(input, boolean_mask_filter<false>{*device_boolean_mask}, stream, mr);
  }
}

}  // namespace detail

/*
 * Filters a table_view using a column_view of boolean values as a mask.
 */
std::unique_ptr<table> apply_boolean_mask(table_view const& input,
                                          column_view const& boolean_mask,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_boolean_mask(input, boolean_mask, stream, mr);
}
}  // namespace cudf
