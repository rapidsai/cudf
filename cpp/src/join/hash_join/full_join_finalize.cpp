/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/join_common_utils.hpp"

#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join_finalize(
  cudf::host_span<cudf::device_span<size_type const> const> left_partials,
  cudf::host_span<cudf::device_span<size_type const> const> right_partials,
  size_type probe_table_num_rows,
  size_type build_table_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return cudf::detail::finalize_full_join(
    left_partials, right_partials, probe_table_num_rows, build_table_num_rows, stream, mr);
}

}  // namespace cudf
