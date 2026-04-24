/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "insert.cuh"

namespace cudf::groupby {

template streaming_groupby::impl::batch_insert_result
streaming_groupby::impl::probe_and_insert_impl<true>(table_view const& batch_keys,
                                                     rmm::cuda_stream_view stream);

}  // namespace cudf::groupby
