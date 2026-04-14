/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "count_kernels.hpp"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"

#include <cudf/detail/iterator.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/fill.h>
#include <thrust/transform.h>

namespace cudf::detail {

std::unique_ptr<rmm::device_uvector<size_type>> make_join_match_counts(
  table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  cudf::detail::hash_table_t const& hash_table,
  bool is_empty,
  bool has_nulls,
  null_equality compare_nulls,
  join_kind join,
  table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto match_counts =
    std::make_unique<rmm::device_uvector<size_type>>(probe.num_rows(), stream, mr);

  if (is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 match_counts->begin(),
                 match_counts->end(),
                 join == join_kind::INNER_JOIN ? 0 : 1);
    return match_counts;
  }

  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  auto const probe_table_num_rows = probe.num_rows();

  auto count_matches = [&](auto equality, auto d_hasher) {
    // Precompute probe keys: {hash(row_idx), row_idx} for each probe row.
    auto const n = static_cast<cuda::std::int64_t>(probe_table_num_rows);
    rmm::device_uvector<probe_key_type> probe_keys(n, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(probe_table_num_rows),
                      probe_keys.begin(),
                      pair_fn{d_hasher});

    auto const ref = hash_table.ref(cuco::op::count)
                       .rebind_key_eq(equality)
                       .rebind_hash_function(hash_table.hash_function());
    if (join == join_kind::INNER_JOIN) {
      launch_count_each<false>(probe_keys.data(), n, match_counts->begin(), ref, stream);
    } else {
      // IsOuter=true handles the clamp (zero → 1) for LEFT/FULL joins internally.
      launch_count_each<true>(probe_keys.data(), n, match_counts->begin(), ref, stream);
    }
  };

  dispatch_join_comparator(
    build, probe, preprocessed_build, preprocessed_probe, has_nulls, compare_nulls, count_matches);

  return match_counts;
}

template <typename Hasher>
std::unique_ptr<rmm::device_uvector<size_type>> hash_join<Hasher>::make_match_counts(
  join_kind join,
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return make_join_match_counts(_build,
                                _preprocessed_build,
                                _impl->_hash_table,
                                _is_empty,
                                _has_nulls,
                                _nulls_equal,
                                join,
                                probe,
                                stream,
                                mr);
}

template std::unique_ptr<rmm::device_uvector<size_type>>
hash_join<hash_join_hasher>::make_match_counts(join_kind,
                                               cudf::table_view const&,
                                               rmm::cuda_stream_view,
                                               rmm::device_async_resource_ref) const;

}  // namespace cudf::detail
