/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/types.hpp>

namespace cudf::groupby::detail::hash {

/**
 * @brief Dispatches an aggregation::Kind to a functor, but only for the subset of aggregation kinds
 * supported by the shared-memory groupby path: SUM, SUM_OF_SQUARES, PRODUCT, MIN, MAX,
 * COUNT_VALID, COUNT_ALL, ARGMIN, ARGMAX, STD, VARIANCE.
 *
 * This avoids instantiating templates for the ~25 unsupported kinds, significantly reducing compile
 * time for TUs that only need the shared-memory aggregation path.
 */
template <typename F, typename... Ts>
CUDF_HOST_DEVICE inline decltype(auto) shmem_aggregation_dispatcher(cudf::aggregation::Kind k,
                                                                    F&& f,
                                                                    Ts&&... args)
{
  switch (k) {
    case cudf::aggregation::SUM:
      return f.template operator()<cudf::aggregation::SUM>(std::forward<Ts>(args)...);
    case cudf::aggregation::SUM_OF_SQUARES:
      return f.template operator()<cudf::aggregation::SUM_OF_SQUARES>(std::forward<Ts>(args)...);
    case cudf::aggregation::PRODUCT:
      return f.template operator()<cudf::aggregation::PRODUCT>(std::forward<Ts>(args)...);
    case cudf::aggregation::MIN:
      return f.template operator()<cudf::aggregation::MIN>(std::forward<Ts>(args)...);
    case cudf::aggregation::MAX:
      return f.template operator()<cudf::aggregation::MAX>(std::forward<Ts>(args)...);
    case cudf::aggregation::COUNT_VALID:
      return f.template operator()<cudf::aggregation::COUNT_VALID>(std::forward<Ts>(args)...);
    case cudf::aggregation::COUNT_ALL:
      return f.template operator()<cudf::aggregation::COUNT_ALL>(std::forward<Ts>(args)...);
    case cudf::aggregation::ARGMIN:
      return f.template operator()<cudf::aggregation::ARGMIN>(std::forward<Ts>(args)...);
    case cudf::aggregation::ARGMAX:
      return f.template operator()<cudf::aggregation::ARGMAX>(std::forward<Ts>(args)...);
    case cudf::aggregation::STD:
      return f.template operator()<cudf::aggregation::STD>(std::forward<Ts>(args)...);
    case cudf::aggregation::VARIANCE:
      return f.template operator()<cudf::aggregation::VARIANCE>(std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Unsupported aggregation in shared memory path.");
#else
      CUDF_UNREACHABLE("Unsupported aggregation in shared memory path.");
#endif
    }
  }
}

namespace {
struct shmem_dispatch_source {
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  template <typename Element, typename F, typename... Ts>
  CUDF_HOST_DEVICE inline decltype(auto) operator()(cudf::aggregation::Kind k,
                                                    F&& f,
                                                    Ts&&... args) const
  {
    return shmem_aggregation_dispatcher(k,
                                        cudf::detail::dispatch_aggregation<Element>{},
                                        std::forward<F>(f),
                                        std::forward<Ts>(args)...);
  }
};
}  // namespace

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <typename F, typename... Ts>
CUDF_HOST_DEVICE inline constexpr decltype(auto) dispatch_type_and_shmem_aggregation(
  cudf::data_type type, cudf::aggregation::Kind k, F&& f, Ts&&... args)
{
  return type_dispatcher(
    type, shmem_dispatch_source{}, k, std::forward<F>(f), std::forward<Ts>(args)...);
}

}  // namespace cudf::groupby::detail::hash
