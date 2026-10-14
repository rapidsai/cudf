/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "insert_rows.cuh"

namespace cudf::groupby {

template size_type streaming_groupby::impl::probe_and_insert_rows<false>(
  std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
  cudf::nullate::DYNAMIC has_null,
  bitmask_type const* batch_bitmask,
  size_type batch_size,
  size_type* target_indices,
  size_type* batch_local_indices,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

}  // namespace cudf::groupby
