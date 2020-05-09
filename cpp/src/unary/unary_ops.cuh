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

#ifndef UNARY_OPS_H
#define UNARY_OPS_H

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace experimental {
namespace unary {
template <typename T, typename Tout, typename F>
struct launcher {
  static std::unique_ptr<cudf::column> launch(cudf::column_view const& input,
                                              cudf::experimental::unary_op op,
                                              rmm::mr::device_memory_resource* mr,
                                              cudaStream_t stream = 0)
  {
    std::unique_ptr<cudf::column> output = [&] {
      if (op == cudf::experimental::unary_op::NOT) {
        auto type = cudf::data_type{cudf::BOOL8};
        auto size = input.size();

        return std::make_unique<column>(type,
                                        size,
                                        rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                        cudf::detail::copy_bitmask(input, stream, mr),
                                        input.null_count());

      } else {
        return cudf::experimental::detail::allocate_like(
          input, input.size(), experimental::mask_allocation_policy::RETAIN, stream, mr);
      }
    }();

    if (input.size() == 0) return output;

    auto output_view = output->mutable_view();

    CUDF_EXPECTS(input.size() > 0, "Launcher requires input size to be non-zero.");
    CUDF_EXPECTS(input.size() == output_view.size(),
                 "Launcher requires input and output size to be equal.");

    if (input.nullable())
      output->set_null_mask(
        rmm::device_buffer{input.null_mask(), bitmask_allocation_size_bytes(input.size())},
        input.null_count());

    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      input.begin<T>(),
                      input.end<T>(),
                      output_view.begin<Tout>(),
                      F{});

    CHECK_CUDA(stream);

    return output;
  }
};

}  // namespace unary
}  // namespace experimental
}  // namespace cudf

#endif  // UNARY_OPS_H
