/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/atomic>

namespace cudf::groupby::detail::hash {

template <typename Source>
__device__ constexpr static bool is_m2_var_supported()
{
  return is_numeric<Source>() && !is_fixed_point<Source>();
}

struct m2_hash_functor {
  size_type const* d_output_index_map;
  bitmask_type const* d_row_bitmask;
  mutable_column_device_view target;
  column_device_view source;
  column_device_view sum;
  column_device_view count;
  m2_hash_functor(size_type const* d_output_index_map,
                  bitmask_type const* d_row_bitmask,
                  mutable_column_device_view target,
                  column_device_view source,
                  column_device_view sum,
                  column_device_view count)
    : d_output_index_map{d_output_index_map},
      d_row_bitmask{d_row_bitmask},
      target{target},
      source{source},
      sum{sum},
      count{count}
  {
  }

  template <typename Source>
  __device__ void operator()(column_device_view const&, size_type, size_type) noexcept
    requires(!is_m2_var_supported<Source>())
  {
    CUDF_UNREACHABLE("Invalid source type for M2 aggregation.");
  }

  template <typename Source>
  __device__ void operator()(column_device_view const& source,
                             size_type source_index,
                             size_type target_index) noexcept
    requires(is_m2_var_supported<Source>())
  {
    using Target    = cudf::detail::target_type_t<Source, aggregation::M2>;
    using SumType   = cudf::detail::target_type_t<Source, aggregation::SUM>;
    using CountType = cudf::detail::target_type_t<Source, aggregation::COUNT_VALID>;

    if (source.is_null(source_index)) { return; }
    auto const group_size = count.element<CountType>(target_index);
    if (group_size == 0) { return; }

    auto const x      = static_cast<Target>(source.element<Source>(source_index));
    auto const mean   = static_cast<Target>(sum.element<SumType>(target_index)) / group_size;
    auto const diff   = x - mean;
    auto const result = diff * diff;
    cuda::atomic_ref<Target, cuda::thread_scope_device> ref{target.element<Target>(target_index)};
    ref.fetch_add(result, cuda::std::memory_order_relaxed);
    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }

  __device__ inline void operator()(size_type source_index)
  {
    if (d_row_bitmask == nullptr or bit_is_set(d_row_bitmask, source_index)) {
      auto const target_index = d_output_index_map[source_index];

      auto col         = source;
      auto source_type = source.type();
      if (source_type.id() == type_id::DICTIONARY32) {
        col          = source.child(cudf::dictionary_column_view::keys_column_index);
        source_type  = col.type();
        source_index = static_cast<size_type>(source.element<dictionary32>(source_index));
      }

      type_dispatcher(source_type, *this, col, source_index, target_index);
    }
  }
};

struct var_hash_functor {
  size_type const* d_output_index_map;
  bitmask_type const* d_row_bitmask;
  mutable_column_device_view target;
  column_device_view source;
  column_device_view sum;
  column_device_view count;
  size_type ddof;
  var_hash_functor(size_type const* d_output_index_map,
                   bitmask_type const* d_row_bitmask,
                   mutable_column_device_view target,
                   column_device_view source,
                   column_device_view sum,
                   column_device_view count,
                   size_type ddof)
    : d_output_index_map{d_output_index_map},
      d_row_bitmask{d_row_bitmask},
      target{target},
      source{source},
      sum{sum},
      count{count},
      ddof{ddof}
  {
  }

  template <typename Source>
  __device__ void operator()(column_device_view const&, size_type, size_type) noexcept
    requires(!is_m2_var_supported<Source>())
  {
    CUDF_UNREACHABLE("Invalid source type for std, var aggregation combination.");
  }

  template <typename Source>
  __device__ void operator()(column_device_view const& source,
                             size_type source_index,
                             size_type target_index) noexcept
    requires(is_m2_var_supported<Source>())
  {
    using Target    = cudf::detail::target_type_t<Source, aggregation::VARIANCE>;
    using SumType   = cudf::detail::target_type_t<Source, aggregation::SUM>;
    using CountType = cudf::detail::target_type_t<Source, aggregation::COUNT_VALID>;

    if (source.is_null(source_index)) return;
    CountType group_size = count.element<CountType>(target_index);
    if (group_size == 0 or group_size - ddof <= 0) return;

    auto x        = static_cast<Target>(source.element<Source>(source_index));
    auto mean     = static_cast<Target>(sum.element<SumType>(target_index)) / group_size;
    Target result = (x - mean) * (x - mean) / (group_size - ddof);
    cuda::atomic_ref<Target, cuda::thread_scope_device> ref{target.element<Target>(target_index)};
    ref.fetch_add(result, cuda::std::memory_order_relaxed);
    // STD sqrt is applied in finalize()

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }

  __device__ inline void operator()(size_type source_index)
  {
    if (d_row_bitmask == nullptr or cudf::bit_is_set(d_row_bitmask, source_index)) {
      auto const target_index = d_output_index_map[source_index];

      auto col         = source;
      auto source_type = source.type();
      if (source_type.id() == type_id::DICTIONARY32) {
        col          = source.child(cudf::dictionary_column_view::keys_column_index);
        source_type  = col.type();
        source_index = static_cast<size_type>(source.element<dictionary32>(source_index));
      }

      type_dispatcher(source_type, *this, col, source_index, target_index);
    }
  }
};

}  // namespace cudf::groupby::detail::hash
