/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "count_kernels.cuh"
#include "ref_types.cuh"

namespace cudf::detail {

template void launch_count_each<false, primitive_count_ref_t>(probe_key_type const*,
                                                              cuda::std::int64_t,
                                                              size_type*,
                                                              primitive_count_ref_t,
                                                              rmm::cuda_stream_view);

template void launch_count_each<false, nested_count_ref_t>(
  probe_key_type const*, cuda::std::int64_t, size_type*, nested_count_ref_t, rmm::cuda_stream_view);

template void launch_count_each<false, flat_count_ref_t>(
  probe_key_type const*, cuda::std::int64_t, size_type*, flat_count_ref_t, rmm::cuda_stream_view);

}  // namespace cudf::detail
