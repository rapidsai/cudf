/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ref_types.cuh"
#include "retrieve_kernels.cuh"

namespace cudf::detail {

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
launch_retrieve<true, primitive_count_ref_t>(probe_key_type const*,
                                             cuda::std::int64_t,
                                             size_type const*,
                                             primitive_count_ref_t,
                                             rmm::cuda_stream_view,
                                             rmm::device_async_resource_ref);

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
launch_retrieve<true, nested_count_ref_t>(probe_key_type const*,
                                          cuda::std::int64_t,
                                          size_type const*,
                                          nested_count_ref_t,
                                          rmm::cuda_stream_view,
                                          rmm::device_async_resource_ref);

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
launch_retrieve<true, flat_count_ref_t>(probe_key_type const*,
                                        cuda::std::int64_t,
                                        size_type const*,
                                        flat_count_ref_t,
                                        rmm::cuda_stream_view,
                                        rmm::device_async_resource_ref);

}  // namespace cudf::detail
