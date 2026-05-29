/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace cudf::detail {

/**
 * @brief Implementation of `cudf::streaming_hash_join`.
 *
 * For the initial scope this class supports a single `insert()` call and delegates probing to an
 * internally-constructed `cudf::hash_join`. Multi-partition support (n-table row equality
 * dispatch on probe) will be added in a follow-up; calling `insert()` more than once currently
 * throws.
 */
class streaming_hash_join {
 public:
  streaming_hash_join(host_span<data_type const> right_schema,
                      host_span<size_type const> right_key_indices,
                      size_type total_right_rows,
                      nullable_join has_nulls,
                      null_equality compare_nulls,
                      double load_factor);

  ~streaming_hash_join();
  streaming_hash_join(streaming_hash_join const&)            = delete;
  streaming_hash_join& operator=(streaming_hash_join const&) = delete;
  streaming_hash_join(streaming_hash_join&&) noexcept;
  streaming_hash_join& operator=(streaming_hash_join&&) noexcept;

  void insert(cudf::table_view const& right_partition, rmm::cuda_stream_view stream);

  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& left,
             std::optional<std::size_t> output_size,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr) const;

 private:
  std::vector<data_type> _right_schema;
  std::vector<size_type> _right_key_indices;
  size_type _total_right_rows;
  nullable_join _has_nulls;
  null_equality _compare_nulls;
  double _load_factor;

  size_type _inserted_rows{0};
  std::unique_ptr<cudf::hash_join> _hash_join;
};

}  // namespace cudf::detail
