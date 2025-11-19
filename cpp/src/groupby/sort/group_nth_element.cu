/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_nth_element(column_view const& values,
                                          column_view const& group_sizes,
                                          cudf::device_span<size_type const> group_labels,
                                          cudf::device_span<size_type const> group_offsets,
                                          size_type num_groups,
                                          size_type n,
                                          null_policy null_handling,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  if (num_groups == 0) { return empty_like(values); }

  auto nth_index = rmm::device_uvector<size_type>(num_groups, stream);
  // TODO: replace with async version
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), nth_index.begin(), num_groups, values.size());

  // nulls_policy::INCLUDE (equivalent to pandas nth(dropna=None) but return nulls for n
  if (null_handling == null_policy::INCLUDE || !values.has_nulls()) {
    // Returns index of nth value.
    thrust::transform_if(
      rmm::exec_policy(stream),
      group_sizes.begin<size_type>(),
      group_sizes.end<size_type>(),
      group_offsets.begin(),
      group_sizes.begin<size_type>(),  // stencil
      nth_index.begin(),
      [n] __device__(auto group_size, auto group_offset) {
        return group_offset + ((n < 0) ? group_size + n : n);
      },
      [n] __device__(auto group_size) {  // nth within group
        return (n < 0) ? group_size >= (-n) : group_size > n;
      });
  } else {  // skip nulls (equivalent to pandas nth(dropna='any'))
    // Returns index of nth value.
    auto values_view = column_device_view::create(values, stream);
    auto bitmask_iterator =
      thrust::make_transform_iterator(cudf::detail::make_validity_iterator(*values_view),
                                      cuda::proclaim_return_type<size_type>([] __device__(auto b) {
                                        return static_cast<size_type>(b);
                                      }));
    rmm::device_uvector<size_type> intra_group_index(values.size(), stream);
    // intra group index for valids only.
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                  group_labels.begin(),
                                  group_labels.end(),
                                  bitmask_iterator,
                                  intra_group_index.begin());
    // group_size to recalculate n if n<0
    rmm::device_uvector<size_type> group_count = [&] {
      if (n < 0) {
        rmm::device_uvector<size_type> group_count(num_groups, stream);
        thrust::reduce_by_key(rmm::exec_policy(stream),
                              group_labels.begin(),
                              group_labels.end(),
                              bitmask_iterator,
                              thrust::make_discard_iterator(),
                              group_count.begin());
        return group_count;
      } else {
        return rmm::device_uvector<size_type>(0, stream);
      }
    }();
    // gather the valid index == n
    thrust::scatter_if(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       thrust::make_counting_iterator<size_type>(values.size()),
                       group_labels.begin(),                          // map
                       thrust::make_counting_iterator<size_type>(0),  // stencil
                       nth_index.begin(),
                       [n,
                        bitmask_iterator,
                        group_size        = group_count.begin(),
                        group_labels      = group_labels.begin(),
                        intra_group_index = intra_group_index.begin()] __device__(auto i) -> bool {
                         auto nth = ((n < 0) ? group_size[group_labels[i]] + n : n);
                         return (bitmask_iterator[i] && intra_group_index[i] == nth);
                       });
  }

  auto output_table = cudf::detail::gather(table_view{{values}},
                                           nth_index,
                                           out_of_bounds_policy::NULLIFY,
                                           cudf::detail::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           mr);
  if (!output_table->get_column(0).has_nulls()) output_table->get_column(0).set_null_mask({}, 0);
  return std::make_unique<column>(std::move(output_table->get_column(0)));
}
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
