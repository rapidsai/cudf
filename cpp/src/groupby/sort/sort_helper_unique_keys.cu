/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/gather.cuh>
#include <cudf/detail/groupby/sort_helper.hpp>

#include <cuda/functional>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace sort {

std::unique_ptr<table> sort_groupby_helper::unique_keys(rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  auto idx_data = key_sort_order(stream).data<size_type>();

  auto gather_map_it =
    thrust::make_transform_iterator(group_offsets(stream).begin(),
                                    cuda::proclaim_return_type<size_type>(
                                      [idx_data] __device__(size_type i) { return idx_data[i]; }));

  return cudf::detail::gather(_keys,
                              gather_map_it,
                              gather_map_it + num_groups(stream),
                              out_of_bounds_policy::DONT_CHECK,
                              stream,
                              mr);
}

}  // namespace sort
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
