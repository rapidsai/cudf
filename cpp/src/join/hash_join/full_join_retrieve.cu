/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../join_common_utils.hpp"
#include "retrieve_common.cuh"

#include <cudf/detail/nvtx/ranges.hpp>

namespace cudf::detail {

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::full_join(cudf::table_view const& probe,
                             std::optional<std::size_t> output_size,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  validate_hash_join_probe(_build, probe, _has_nulls);

  if (_is_empty) { return get_trivial_left_join_indices(probe, stream, mr); }

  if (is_trivial_join(probe, _build, join_kind::FULL_JOIN)) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);

  auto join_indices = cudf::detail::probe_join_hash_table<join_kind::FULL_JOIN>(_build,
                                                                                probe,
                                                                                _preprocessed_build,
                                                                                preprocessed_probe,
                                                                                _hash_table,
                                                                                _has_nulls,
                                                                                _nulls_equal,
                                                                                output_size,
                                                                                stream,
                                                                                mr);

  auto complement_indices = detail::get_left_join_indices_complement(
    join_indices.second, probe.num_rows(), _build.num_rows(), stream, mr);
  return detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::full_join(cudf::table_view const& probe,
                                       std::optional<std::size_t> output_size,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail
