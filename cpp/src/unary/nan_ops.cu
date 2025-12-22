/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
struct nan_dispatcher {
  template <typename T, typename Predicate>
  std::unique_ptr<column> operator()(cudf::column_view const& input,
                                     Predicate predicate,
                                     rmm::cuda_stream_view stream,
                                     cudf::memory_resources resources)
    requires(std::is_floating_point_v<T>)
  {
    auto input_device_view = column_device_view::create(input, stream);

    if (input.has_nulls()) {
      auto input_pair_iterator = make_pair_iterator<T, true>(*input_device_view);
      return true_if(input_pair_iterator,
                     input_pair_iterator + input.size(),
                     input.size(),
                     predicate,
                     stream,
                     resources);
    } else {
      auto input_pair_iterator = make_pair_iterator<T, false>(*input_device_view);
      return true_if(input_pair_iterator,
                     input_pair_iterator + input.size(),
                     input.size(),
                     predicate,
                     stream,
                     resources);
    }
  }

  template <typename T, typename Predicate>
  std::unique_ptr<column> operator()(cudf::column_view const& input,
                                     Predicate predicate,
                                     rmm::cuda_stream_view stream,
                                     cudf::memory_resources resources)
    requires(!std::is_floating_point_v<T>)
  {
    CUDF_FAIL("NAN is not supported in a Non-floating point type column");
  }
};

std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               rmm::cuda_stream_view stream,
                               cudf::memory_resources resources)
{
  auto predicate = [] __device__(auto element_validity_pair) {
    return element_validity_pair.second and std::isnan(element_validity_pair.first);
  };

  return cudf::type_dispatcher(input.type(), nan_dispatcher{}, input, predicate, stream, resources);
}

std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                   rmm::cuda_stream_view stream,
                                   cudf::memory_resources resources)
{
  auto predicate = [] __device__(auto element_validity_pair) {
    return !element_validity_pair.second or !std::isnan(element_validity_pair.first);
  };

  return cudf::type_dispatcher(input.type(), nan_dispatcher{}, input, predicate, stream, resources);
}

}  // namespace detail

std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               rmm::cuda_stream_view stream,
                               cudf::memory_resources resources)
{
  CUDF_FUNC_RANGE();
  return detail::is_nan(input, stream, resources);
}

std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                   rmm::cuda_stream_view stream,
                                   cudf::memory_resources resources)
{
  CUDF_FUNC_RANGE();
  return detail::is_not_nan(input, stream, resources);
}

}  // namespace cudf
