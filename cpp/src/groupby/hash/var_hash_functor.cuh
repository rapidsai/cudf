/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cuco/static_set_ref.cuh>
#include <cuda/atomic>
#include <cuda/std/type_traits>

namespace cudf::groupby::detail::hash {
template <typename SetType>
struct var_hash_functor {
  SetType set;
  bitmask_type const* __restrict__ row_bitmask;
  mutable_column_device_view target;
  column_device_view source;
  column_device_view sum;
  column_device_view count;
  size_type ddof;
  var_hash_functor(SetType set,
                   bitmask_type const* row_bitmask,
                   mutable_column_device_view target,
                   column_device_view source,
                   column_device_view sum,
                   column_device_view count,
                   size_type ddof)
    : set{set},
      row_bitmask{row_bitmask},
      target{target},
      source{source},
      sum{sum},
      count{count},
      ddof{ddof}
  {
  }

  template <typename Source>
  constexpr static bool is_supported()
  {
    return is_numeric<Source>() && !is_fixed_point<Source>();
  }

  template <typename Source>
  __device__ cuda::std::enable_if_t<!is_supported<Source>()> operator()(
    column_device_view const& source, size_type source_index, size_type target_index) noexcept
  {
    CUDF_UNREACHABLE("Invalid source type for std, var aggregation combination.");
  }

  template <typename Source>
  __device__ cuda::std::enable_if_t<is_supported<Source>()> operator()(
    column_device_view const& source, size_type source_index, size_type target_index) noexcept
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
    if (row_bitmask == nullptr or cudf::bit_is_set(row_bitmask, source_index)) {
      auto const target_index = *set.find(source_index);

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
