/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <optional>

namespace cudf::lists {
namespace detail {
namespace {
template <typename T>
struct tabulator {
  size_type const n_lists;
  size_type const n_elements;

  T const* const starts;
  T const* const steps;
  offset_type const* const offsets;

  template <typename U>
  static std::enable_if_t<!cudf::is_duration<U>(), T> __device__ multiply(U x, size_type times)
  {
    return x * static_cast<T>(times);
  }

  template <typename U>
  static std::enable_if_t<cudf::is_duration<U>(), T> __device__ multiply(U x, size_type times)
  {
    return T{x.count() * times};
  }

  auto __device__ operator()(size_type idx) const
  {
    auto const list_idx_end = thrust::upper_bound(thrust::seq, offsets, offsets + n_lists, idx);
    auto const list_idx     = thrust::distance(offsets, list_idx_end) - 1;
    auto const list_offset  = offsets[list_idx];
    auto const list_step    = steps ? steps[list_idx] : T{1};
    return starts[list_idx] + multiply(list_step, idx - list_offset);
  }
};

template <typename T, typename Enable = void>
struct sequences_functor {
  template <typename... Args>
  static std::unique_ptr<column> invoke(Args&&...)
  {
    CUDF_FAIL("Unsupported per-list sequence type-agg combination.");
  }
};

struct sequences_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(size_type n_lists,
                                     size_type n_elements,
                                     column_view const& starts,
                                     std::optional<column_view> const& steps,
                                     offset_type const* offsets,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return sequences_functor<T>::invoke(n_lists, n_elements, starts, steps, offsets, stream, mr);
  }
};

template <typename T>
static constexpr bool is_supported()
{
  return (cudf::is_numeric<T>() && !cudf::is_boolean<T>()) || cudf::is_duration<T>();
}

template <typename T>
struct sequences_functor<T, std::enable_if_t<is_supported<T>()>> {
  static std::unique_ptr<column> invoke(size_type n_lists,
                                        size_type n_elements,
                                        column_view const& starts,
                                        std::optional<column_view> const& steps,
                                        offset_type const* offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    auto result =
      make_fixed_width_column(starts.type(), n_elements, mask_state::UNALLOCATED, stream, mr);
    if (starts.is_empty()) { return result; }

    auto const result_begin = result->mutable_view().template begin<T>();

    // Use pointers instead of column_device_view to access start and step values should be enough.
    // This is because we don't need to check for nulls and only support numeric and duration types.
    auto const starts_begin = starts.template begin<T>();
    auto const steps_begin  = steps ? steps.value().template begin<T>() : nullptr;

    auto const op = tabulator<T>{n_lists, n_elements, starts_begin, steps_begin, offsets};
    thrust::tabulate(rmm::exec_policy(stream), result_begin, result_begin + n_elements, op);

    return result;
  }
};

std::unique_ptr<column> make_empty_lists_column(data_type child_type,
                                                rmm::cuda_stream_view stream,
                                                rmm::mr::device_memory_resource* mr)
{
  auto offsets = make_empty_column(data_type(type_to_id<offset_type>()));
  auto child   = make_empty_column(child_type);
  return make_lists_column(
    0, std::move(offsets), std::move(child), 0, rmm::device_buffer(0, stream, mr), stream, mr);
}

std::unique_ptr<column> sequences(column_view const& starts,
                                  std::optional<column_view> const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!starts.has_nulls() && !sizes.has_nulls(),
               "starts and sizes input columns must not have nulls.");
  CUDF_EXPECTS(starts.size() == sizes.size(),
               "starts and sizes input columns must have the same number of rows.");
  CUDF_EXPECTS(cudf::is_index_type(sizes.type()), "Input sizes column must be of integer types.");

  if (steps) {
    auto const& steps_cv = steps.value();
    CUDF_EXPECTS(!steps_cv.has_nulls(), "steps input column must not have nulls.");
    CUDF_EXPECTS(starts.size() == steps_cv.size(),
                 "starts and steps input columns must have the same number of rows.");
    CUDF_EXPECTS(starts.type() == steps_cv.type(),
                 "starts and steps input columns must have the same type.");
  }

  auto const n_lists = starts.size();
  if (n_lists == 0) { return make_empty_lists_column(starts.type(), stream, mr); }

  // Generate list offsets for the output.
  auto list_offsets = make_numeric_column(
    data_type(type_to_id<offset_type>()), n_lists + 1, mask_state::UNALLOCATED, stream, mr);
  auto const offsets_begin  = list_offsets->mutable_view().template begin<offset_type>();
  auto const sizes_input_it = cudf::detail::indexalator_factory::make_input_iterator(sizes);

  thrust::exclusive_scan(
    rmm::exec_policy(stream), sizes_input_it, sizes_input_it + n_lists + 1, offsets_begin);
  auto const n_elements = cudf::detail::get_value<size_type>(list_offsets->view(), n_lists, stream);

  auto child = type_dispatcher(starts.type(),
                               sequences_dispatcher{},
                               n_lists,
                               n_elements,
                               starts,
                               steps,
                               offsets_begin,
                               stream,
                               mr);

  return make_lists_column(n_lists,
                           std::move(list_offsets),
                           std::move(child),
                           0,
                           rmm::device_buffer(0, stream, mr),
                           stream,
                           mr);
}

}  // anonymous namespace

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  return sequences(starts, std::nullopt, sizes, stream, mr);
}

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  return sequences(starts, std::optional<column_view>{steps}, sizes, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& sizes,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sequences(starts, sizes, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& steps,
                                  column_view const& sizes,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sequences(starts, steps, sizes, cudf::get_default_stream(), mr);
}

}  // namespace cudf::lists
