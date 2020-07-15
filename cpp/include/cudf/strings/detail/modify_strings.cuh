/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <strings/utilities.cuh>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Generic string modification in two passes: 1st pass probes for memory load requirements;
 * 2nd pass executes string modification.
 *
 * @tparam device_probe_functor Functor for probing memory requirements;
 * must implement `__device__ int32_t operator()(size_type idx) const`
 * @tparam device_execute_functor Functor for executing string modification; must
 * implement `__device__ int32_t operator()(size_type idx)`
 * @tparam ...Types Types of possible additional arguments to be forwarded
 * to the probe / execute functors (pre-condition: must both take the same trailling pack of
 * arguments, in addition to their required args)
 *
 * @param strings Number Column of strings to apply the modifications on;
 * it is not modified in place; rather a new column is returned instead
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * (cannot be a default argument because of the variadic pack);
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * (cannot be a default argument because of the variadic pack);
 * @param ...args Additional arguments to be forwarded to
 * the probe / execute constructors (can be empty);
 * @return modified strings column
 */
template <typename device_probe_functor, typename device_execute_functor, typename... Types>
std::unique_ptr<column> modify_strings(strings_column_view const& strings,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream,
                                       Types&&... args)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return detail::make_empty_strings_column(mr, stream);

  auto execpol = rmm::exec_policy(stream);

  auto strings_column  = column_device_view::create(strings.parent(), stream);
  auto d_column        = *strings_column;
  size_type null_count = strings.null_count();

  // copy null mask
  rmm::device_buffer null_mask = copy_bitmask(strings.parent(), stream, mr);
  // get the lookup tables used for case conversion

  device_probe_functor d_probe_fctr{d_column, std::forward<Types>(args)...};

  // build offsets column -- calculate the size of each output string
  auto offsets_transformer_itr =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), d_probe_fctr);
  auto offsets_column = detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto offsets_view = offsets_column->view();
  auto d_new_offsets =
    offsets_view.template data<int32_t>();  // not sure why this requires `.template` and the next
                                            // one (`d_chars = ...`) doesn't

  // build the chars column -- convert characters based on case_flag parameter
  size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
  auto chars_column =
    strings::detail::create_chars_child_column(strings_count, null_count, bytes, mr, stream);
  auto chars_view = chars_column->mutable_view();
  auto d_chars    = chars_view.data<char>();

  device_execute_functor d_execute_fctr{
    d_column, d_new_offsets, d_chars, std::forward<Types>(args)...};

  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     d_execute_fctr);

  //
  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
