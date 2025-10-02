
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/rtc/cache.hpp"

#include <cudf/detail/nvtx/ranges.hpp>

namespace cudf {
namespace rtc {

void cache_t::store_blob(sha256_hash const& sha, blob binary) { CUDF_FUNC_RANGE(); }

blob cache_t::query_blob(sha256_hash const& sha) { CUDF_FUNC_RANGE(); }

void cache_t::store_fragment(sha256_hash const& sha, fragment frag) { CUDF_FUNC_RANGE(); }

fragment cache_t::query_fragment(sha256_hash const& sha) { CUDF_FUNC_RANGE(); }

void cache_t::store_module(sha256_hash const& sha, module mod) { CUDF_FUNC_RANGE(); }

module cache_t::query_module(sha256_hash const& sha) { CUDF_FUNC_RANGE(); }

}  // namespace rtc
}  // namespace cudf