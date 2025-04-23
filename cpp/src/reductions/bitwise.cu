/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "simple.cuh"

namespace cudf::reduction::detail {

namespace {

/**
 * @brief A map from cudf::type_id to cudf type that excludes non-integral types.
 *
 * This is needed because `same_element_type_dispatcher` executes the same code path for all
 * non-dictionary non-nested non-fixed-point types.
 */
template <type_id t>
struct dispatch_void_if_non_integral {
  using type = std::conditional_t<std::is_integral_v<id_to_type<t>>, id_to_type<t>, void>;
};

}  // namespace

std::unique_ptr<scalar> bitwise_reduction(bitwise_op bit_op,
                                          column_view const& col,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const dtype =
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
  auto const do_reduction = [&](auto const op) {
    return cudf::type_dispatcher<dispatch_void_if_non_integral>(
      dtype, op, col, std::nullopt, stream, mr);
  };

  switch (bit_op) {
    case bitwise_op::AND: {
      auto const op = simple::detail::same_element_type_dispatcher<op::bit_and>{};
      return do_reduction(op);
    }
    case bitwise_op::OR: {
      auto const op = simple::detail::same_element_type_dispatcher<op::bit_or>{};
      return do_reduction(op);
    }
    case bitwise_op::XOR: {
      auto const op = simple::detail::same_element_type_dispatcher<op::bit_xor>{};
      return do_reduction(op);
    }
    default: CUDF_UNREACHABLE("Unsupported bitwise operation");
  }
}

}  // namespace cudf::reduction::detail
