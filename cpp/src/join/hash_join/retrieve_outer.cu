/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ref_types.cuh"
#include "retrieve_kernels.cuh"

namespace cudf::detail {

template std::size_t launch_retrieve<true, primitive_count_ref_t>(probe_key_type const*,
                                                                  cuda::std::int64_t,
                                                                  size_type*,
                                                                  size_type*,
                                                                  size_type const*,
                                                                  primitive_count_ref_t,
                                                                  rmm::cuda_stream_view);

template std::size_t launch_retrieve<true, nested_count_ref_t>(probe_key_type const*,
                                                               cuda::std::int64_t,
                                                               size_type*,
                                                               size_type*,
                                                               size_type const*,
                                                               nested_count_ref_t,
                                                               rmm::cuda_stream_view);

template std::size_t launch_retrieve<true, flat_count_ref_t>(probe_key_type const*,
                                                             cuda::std::int64_t,
                                                             size_type*,
                                                             size_type*,
                                                             size_type const*,
                                                             flat_count_ref_t,
                                                             rmm::cuda_stream_view);

}  // namespace cudf::detail
