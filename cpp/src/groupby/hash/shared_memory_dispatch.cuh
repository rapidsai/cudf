/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace cudf::groupby::detail::hash {
/// Shared memory data alignment
CUDF_HOST_DEVICE cudf::size_type constexpr ALIGNMENT = 16;

// Dictionary and nested value columns are rejected before this kernel is launched.
struct unsupported_shared_memory_type {};

template <cudf::type_id Id>
struct dispatch_shared_memory_type {
  using type = cuda::std::conditional_t<Id == cudf::type_id::DICTIONARY32 or
                                          Id == cudf::type_id::LIST or Id == cudf::type_id::STRUCT,
                                        unsupported_shared_memory_type,
                                        cudf::id_to_type<Id>>;
};

// Compound hash aggregations are decomposed into these simple aggregations before this kernel is
// launched. SUM_OVERFLOW is the only other simple hash aggregation and is explicitly rejected by
// is_shared_memory_compatible.
template <typename F, typename... Ts>
__device__ auto dispatch_shared_memory_aggregation(cudf::aggregation::Kind kind,
                                                   F&& f,
                                                   Ts&&... args)
{
  switch (kind) {
    case cudf::aggregation::SUM:
      return f.template operator()<cudf::aggregation::SUM>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::PRODUCT:
      return f.template operator()<cudf::aggregation::PRODUCT>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::MIN:
      return f.template operator()<cudf::aggregation::MIN>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::MAX:
      return f.template operator()<cudf::aggregation::MAX>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::COUNT_VALID:
      return f.template operator()<cudf::aggregation::COUNT_VALID>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::COUNT_ALL:
      return f.template operator()<cudf::aggregation::COUNT_ALL>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::SUM_OF_SQUARES:
      return f.template operator()<cudf::aggregation::SUM_OF_SQUARES>(
        cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::ARGMAX:
      return f.template operator()<cudf::aggregation::ARGMAX>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::ARGMIN:
      return f.template operator()<cudf::aggregation::ARGMIN>(cuda::std::forward<Ts>(args)...);
    default: CUDF_UNREACHABLE("Unsupported shared memory aggregation.");
  }
}

template <typename Element>
struct dispatch_shared_memory_aggregation_fn {
  template <cudf::aggregation::Kind kind, typename F, typename... Ts>
  __device__ auto operator()(F&& f, Ts&&... args) const
  {
    return f.template operator()<Element, kind>(cuda::std::forward<Ts>(args)...);
  }
};

struct dispatch_shared_memory_source_fn {
  template <typename Element, typename F, typename... Ts>
  __device__ auto operator()(cudf::aggregation::Kind kind, F&& f, Ts&&... args) const
  {
    if constexpr (cuda::std::is_same_v<Element, unsupported_shared_memory_type>) {
      CUDF_UNREACHABLE("Unsupported shared memory aggregation type.");
    } else {
      return dispatch_shared_memory_aggregation(kind,
                                                dispatch_shared_memory_aggregation_fn<Element>{},
                                                cuda::std::forward<F>(f),
                                                cuda::std::forward<Ts>(args)...);
    }
  }
};

template <typename F, typename... Ts>
__device__ auto dispatch_shared_memory_type_and_aggregation(cudf::data_type type,
                                                            cudf::aggregation::Kind kind,
                                                            F&& f,
                                                            Ts&&... args)
{
  return cudf::type_dispatcher<dispatch_shared_memory_type>(type,
                                                            dispatch_shared_memory_source_fn{},
                                                            kind,
                                                            cuda::std::forward<F>(f),
                                                            cuda::std::forward<Ts>(args)...);
}

}  // namespace cudf::groupby::detail::hash
