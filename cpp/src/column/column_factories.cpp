/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {
namespace {
struct size_of_helper {
  template <typename T>
  constexpr int operator()() const noexcept {
    return sizeof(T);
  }
};
}  // namespace

std::size_t size_of(data_type element_type) {
  CUDF_EXPECTS(is_fixed_width(element_type), "Invalid element type.");
  return cudf::exp::type_dispatcher(element_type, size_of_helper{});
}

// Allocate storage for a specified number of numeric elements
std::unique_ptr<column> make_numeric_column(
    data_type type, size_type size, mask_state state, cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");

  return std::make_unique<column>(
      type, size, rmm::device_buffer{size * cudf::size_of(type), stream, mr},
      create_null_mask(size, state, stream, mr), state_null_count(state, size),
      std::vector<std::unique_ptr<column>>{});
}
}  // namespace cudf
