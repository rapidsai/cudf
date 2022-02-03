/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "stream_compaction_common.hpp"

namespace cudf {
namespace detail {

/**
 * @brief Device callable to hash a given row.
 */
template <typename Nullate>
class compaction_hash {
 public:
  compaction_hash(Nullate has_nulls, table_device_view t) : _hash{has_nulls, t} {}

  __device__ inline auto operator()(size_type i) const noexcept
  {
    auto hash = _hash(i);
    return (hash == COMPACTION_EMPTY_KEY_SENTINEL) ? (hash - 1) : hash;
  }

 private:
  row_hash _hash;
};

/**
￼ * @brief Device functor to determine if a row is valid.
￼ */
class row_validity {
 public:
  row_validity(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ inline bool operator()(const size_type& i) const noexcept
  {
    return cudf::bit_is_set(_row_bitmask, i);
  }

 private:
  bitmask_type const* _row_bitmask;
};

}  // namespace detail
}  // namespace cudf
