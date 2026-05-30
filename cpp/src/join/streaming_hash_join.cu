/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/join/streaming_hash_join.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <stdexcept>

namespace cudf {
namespace detail {

streaming_hash_join::streaming_hash_join(host_span<data_type const> right_schema,
                                         host_span<size_type const> right_key_indices,
                                         size_type total_right_rows,
                                         nullable_join has_nulls,
                                         null_equality compare_nulls,
                                         double load_factor)
  : _right_schema{right_schema.begin(), right_schema.end()},
    _right_key_indices{right_key_indices.begin(), right_key_indices.end()},
    _total_right_rows{total_right_rows},
    _has_nulls{has_nulls},
    _compare_nulls{compare_nulls},
    _load_factor{load_factor}
{
  CUDF_EXPECTS(!_right_schema.empty(),
               "streaming_hash_join requires at least one right-side column.",
               std::invalid_argument);
  CUDF_EXPECTS(!_right_key_indices.empty(),
               "streaming_hash_join requires at least one right-side key column.",
               std::invalid_argument);
  CUDF_EXPECTS(_total_right_rows >= 0,
               "streaming_hash_join requires total_right_rows >= 0.",
               std::invalid_argument);
  CUDF_EXPECTS(load_factor > 0.0 && load_factor <= 1.0,
               "streaming_hash_join requires load_factor in (0, 1].",
               std::invalid_argument);
  auto const schema_size = static_cast<size_type>(_right_schema.size());
  for (auto const idx : _right_key_indices) {
    CUDF_EXPECTS(idx >= 0 && idx < schema_size,
                 "streaming_hash_join key index is out of range for the provided schema.",
                 std::invalid_argument);
  }
}

streaming_hash_join::~streaming_hash_join()                                         = default;
streaming_hash_join::streaming_hash_join(streaming_hash_join&&) noexcept            = default;
streaming_hash_join& streaming_hash_join::operator=(streaming_hash_join&&) noexcept = default;

void streaming_hash_join::insert(cudf::table_view const& right_partition,
                                 rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(static_cast<size_type>(right_partition.num_columns()) ==
                 static_cast<size_type>(_right_schema.size()),
               "streaming_hash_join: inserted partition column count does not match schema.",
               std::invalid_argument);
  for (size_type i = 0; i < right_partition.num_columns(); ++i) {
    CUDF_EXPECTS(right_partition.column(i).type() == _right_schema[i],
                 "streaming_hash_join: inserted partition column type does not match schema.",
                 std::invalid_argument);
  }
  CUDF_EXPECTS(!_hash_join,
               "streaming_hash_join: multi-partition insert is not yet implemented; insert() may "
               "currently be called at most once.",
               std::invalid_argument);
  CUDF_EXPECTS(_inserted_rows + right_partition.num_rows() <= _total_right_rows,
               "streaming_hash_join: cumulative inserted rows would exceed total_right_rows.",
               std::invalid_argument);

  _hash_join = std::make_unique<cudf::hash_join>(
    right_partition, _has_nulls, _compare_nulls, _load_factor, stream);
  _inserted_rows += right_partition.num_rows();
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                    std::unique_ptr<rmm::device_uvector<size_type>>>>
streaming_hash_join::inner_join(cudf::table_view const& left,
                                std::optional<std::size_t> output_size,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(
    _hash_join, "streaming_hash_join: inner_join called before any insert().", std::logic_error);
  auto [left_indices, right_row_indices] = _hash_join->inner_join(left, output_size, stream, mr);
  // Single-partition scaffold: every match originates from batch 0. The batch index column is
  // sized to the row-index column and filled with zeros on the caller's stream.
  auto right_batch_indices =
    std::make_unique<rmm::device_uvector<size_type>>(right_row_indices->size(), stream, mr);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               right_batch_indices->begin(),
               right_batch_indices->end(),
               size_type{0});
  return std::pair{std::move(left_indices),
                   std::pair{std::move(right_batch_indices), std::move(right_row_indices)}};
}

}  // namespace detail

streaming_hash_join::streaming_hash_join(host_span<data_type const> right_schema,
                                         host_span<size_type const> right_key_indices,
                                         size_type total_right_rows,
                                         nullable_join has_nulls,
                                         null_equality compare_nulls,
                                         double load_factor)
  : _impl{std::make_unique<cudf::detail::streaming_hash_join>(
      right_schema, right_key_indices, total_right_rows, has_nulls, compare_nulls, load_factor)}
{
}

streaming_hash_join::~streaming_hash_join()                                         = default;
streaming_hash_join::streaming_hash_join(streaming_hash_join&&) noexcept            = default;
streaming_hash_join& streaming_hash_join::operator=(streaming_hash_join&&) noexcept = default;

void streaming_hash_join::insert(cudf::table_view const& right_partition,
                                 rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  _impl->insert(right_partition, stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                    std::unique_ptr<rmm::device_uvector<size_type>>>>
streaming_hash_join::inner_join(cudf::table_view const& left,
                                std::optional<std::size_t> output_size,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->inner_join(left, output_size, stream, mr);
}

}  // namespace cudf
