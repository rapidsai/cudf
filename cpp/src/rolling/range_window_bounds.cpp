/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include "range_window_bounds_detail.hpp"

namespace cudf {
namespace {

struct type_deducing_range_scaler {
  template <typename OrderByColumnType>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar,
                                     bool is_unbounded_range,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    return cudf::type_dispatcher(range_scalar.type(),
                                 cudf::detail::range_scaler<OrderByColumnType>{},
                                 range_scalar,
                                 is_unbounded_range,
                                 stream,
                                 mr);
  }
};

}  // namespace

void range_window_bounds::scale_to(data_type target_type,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  scalar const& range_scalar = *_range_scalar;

  _range_scalar = std::move(cudf::type_dispatcher(
    target_type, type_deducing_range_scaler{}, range_scalar, _is_unbounded, stream, mr));
  assert_invariants();
}

void range_window_bounds::assert_invariants() const
{
  CUDF_EXPECTS(_range_scalar.get(), "Range window scalar cannot be null.");
  CUDF_EXPECTS(_is_unbounded || _range_scalar->is_valid(),
               "Bounded Range window scalar must be valid.");
}

range_window_bounds range_window_bounds::unbounded(data_type type)
{
  return range_window_bounds(true, make_default_constructed_scalar(type));
}

}  // namespace cudf
