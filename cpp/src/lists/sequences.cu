/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/tabulate.h>

#include <optional>

namespace cudf::lists {
namespace detail {
namespace {
template <typename T>
struct tabulator {
  column_device_view const starts;
  column_device_view const steps;
  offset_type const* const offsets;
  size_type const* const labels;

  template <typename T_ = T>
  static std::enable_if_t<!cudf::is_duration<T_>(), T> __device__ multiply(T x, size_type times)
  {
    return x * static_cast<T>(times);
  }

  template <typename T_ = T>
  static std::enable_if_t<cudf::is_duration<T_>(), T> __device__ multiply(T x, size_type times)
  {
    return T{x.count() * times};
  }

  auto __device__ operator()(size_type idx) const
  {
    auto const list_idx    = labels[idx] - 1;  // labels are 1-based indices
    auto const list_offset = offsets[list_idx];
    return starts.element<T>(list_idx) + multiply(steps.element<T>(list_idx), idx - list_offset);
  }
};

template <typename T>
struct tabulator_fixed_step {
  column_device_view const starts;
  offset_type const* const offsets;
  size_type const* const labels;

  auto __device__ operator()(size_type idx) const
  {
    auto const list_idx    = labels[idx] - 1;  // labels are 1-based indices
    auto const list_offset = offsets[list_idx];
    return starts.element<T>(list_idx) + static_cast<T>(idx - list_offset);
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
  std::unique_ptr<column> operator()(size_type n_elements,
                                     column_view const& starts,
                                     std::optional<column_view> const& steps,
                                     offset_type const* offsets,
                                     size_type const* labels,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return sequences_functor<T>::invoke(n_elements, starts, steps, offsets, labels, stream, mr);
  }
};

template <typename T>
static constexpr bool is_supported()
{
  return (cudf::is_numeric<T>() && !cudf::is_boolean<T>()) || cudf::is_duration<T>();
}

template <typename T>
struct sequences_functor<T, std::enable_if_t<is_supported<T>()>> {
  static std::unique_ptr<column> invoke(size_type n_elements,
                                        column_view const& starts,
                                        std::optional<column_view> const& steps,
                                        offset_type const* offsets,
                                        size_type const* labels,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    auto result =
      make_fixed_width_column(starts.type(), n_elements, mask_state::UNALLOCATED, stream, mr);
    if (starts.is_empty()) { return result; }

    auto const result_begin  = result->mutable_view().template begin<T>();
    auto const starts_dv_ptr = column_device_view::create(starts, stream);
    auto const steps_dv_ptr  = steps ? column_device_view::create(steps.value(), stream) : nullptr;

    if (steps) {
      auto const op = tabulator<T>{*starts_dv_ptr, *steps_dv_ptr, offsets, labels};
      thrust::tabulate(rmm::exec_policy(stream), result_begin, result_begin + n_elements, op);
    } else {
      auto const op = tabulator_fixed_step<T>{*starts_dv_ptr, offsets, labels};
      thrust::tabulate(rmm::exec_policy(stream), result_begin, result_begin + n_elements, op);
    }

    return result;
  }
};

}  // anonymous namespace

std::unique_ptr<column> sequences(column_view const& starts,
                                  std::optional<column_view> const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_index_type(sizes.type()), "Input sizes column must be of integer types.");
  if (steps) {
    CUDF_EXPECTS(starts.size() == steps.value().size() && starts.size() == sizes.size(),
                 "starts, steps, and sizes input columns must have the same number of rows.");
    CUDF_EXPECTS(starts.type() == steps.value().type(),
                 "starts and steps input columns must have the same type.");
  } else {
    CUDF_EXPECTS(starts.size() == sizes.size(),
                 "starts and sizes input columns must have the same number of rows.");
  }

  auto const n_lists = starts.size();

  // Generate list offsets for the output.
  auto list_offsets = make_numeric_column(
    data_type(type_to_id<offset_type>()), n_lists + 1, mask_state::UNALLOCATED, stream, mr);
  auto const offsets_begin = list_offsets->mutable_view().template begin<offset_type>();

  // Any null of the input columns will result in a null in the output lists column.
  // We need the output null mask early here to normalize the input list sizes.
  auto [null_mask, null_count] =
    steps ? cudf::detail::bitmask_and(table_view{{starts, steps.value(), sizes}}, stream, mr)
          : cudf::detail::bitmask_and(table_view{{starts, sizes}}, stream, mr);

  // Normalize the input sizes:
  // - Convert input integer type into size_type,
  // - Clamp negative sizes to zero, and
  // - Set zero size for null output.
  auto const sizes_input_it = cudf::detail::indexalator_factory::make_input_iterator(sizes);
  auto const sizes_norm_it  = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0),
    [sizes_input_it,
     null_count = null_count,
     bitmask    = static_cast<bitmask_type*>(null_mask.data())] __device__(size_type idx) {
      // Output list size is zero if output bitmask for that list is invalid.
      if (null_count && !cudf::bit_is_set(bitmask, idx)) { return 0; }

      auto const size = sizes_input_it[idx];
      return size < 0 ? 0 : size;
    });

  CUDA_TRY(cudaMemsetAsync(offsets_begin, 0, sizeof(offset_type), stream.value()));
  thrust::inclusive_scan(
    rmm::exec_policy(stream), sizes_norm_it, sizes_norm_it + n_lists, offsets_begin + 1);
  auto const n_elements = cudf::detail::get_value<size_type>(list_offsets->view(), n_lists, stream);

  // Generate (temporary) list labels (1-based list indices) for all elements.
  auto labels = rmm::device_uvector<size_type>(n_elements, stream);
  thrust::upper_bound(rmm::exec_policy(stream),
                      offsets_begin,
                      offsets_begin + n_lists,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n_elements),
                      labels.begin());

  auto child = type_dispatcher(starts.type(),
                               sequences_dispatcher{},
                               n_elements,
                               starts,
                               steps,
                               offsets_begin,
                               labels.begin(),
                               stream,
                               mr);

  return make_lists_column(n_lists,
                           std::move(list_offsets),
                           std::move(child),
                           null_count,
                           std::move(null_mask),
                           stream,
                           mr);
}

}  // namespace detail

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& sizes,
                                  rmm::mr::device_memory_resource* mr)
{
  return detail::sequences(starts, std::nullopt, sizes, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& steps,
                                  column_view const& sizes,
                                  rmm::mr::device_memory_resource* mr)
{
  return detail::sequences(starts, steps, sizes, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
