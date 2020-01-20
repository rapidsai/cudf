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

#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {
namespace detail {
namespace {

struct interleave_columns_functor
{
    template <typename T, typename... Args>
    std::enable_if_t<not cudf::is_fixed_width<T>(), std::unique_ptr<cudf::column>>
    operator()(Args&&... args)
    {
        CUDF_FAIL("Only fixed-width types are supported in interleave_columns.");
    }

    template <typename T>
    std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<cudf::column>>
    operator()(table_view const& input,
               bool create_mask,
               rmm::mr::device_memory_resource *mr,
               cudaStream_t stream = 0)
    {
        auto arch_column = input.column(0);
        auto output_size = input.num_columns() * input.num_rows();
        auto output = allocate_like(arch_column, output_size, mask_allocation_policy::NEVER, mr, stream);
        auto device_input = table_device_view::create(input);
        auto device_output = mutable_column_device_view::create(*output);
        auto index_begin = thrust::make_counting_iterator<size_type>(0);
        auto index_end = thrust::make_counting_iterator<size_type>(output_size);

        auto func_value = [input=*device_input, divisor=input.num_columns()]
            __device__ (size_type idx) {
                return input.column(idx % divisor).element<T>(idx / divisor);
            };

        if (not create_mask)
        {
            thrust::transform(rmm::exec_policy(stream)->on(stream),
                              index_begin,
                              index_end,
                              device_output->data<T>(),
                              func_value);

            return output;
        }

        auto func_validity = [input=*device_input, divisor=input.num_columns()]
            __device__ (size_type idx) {
                return input.column(idx % divisor).is_valid(idx / divisor);
            };

        thrust::transform_if(rmm::exec_policy(stream)->on(stream),
                             index_begin,
                             index_end,
                             device_output->data<T>(),
                             func_value,
                             func_validity);

        rmm::device_buffer mask;
        size_type null_count;

        std::tie(mask, null_count) = valid_if(index_begin,
                                              index_end,
                                              func_validity,
                                              stream,
                                              mr);

        output->set_null_mask(std::move(mask), null_count);

        return output;
    }
};

} // anonymous namespace
} // namespace detail

std::unique_ptr<column>
interleave_columns(table_view const& input,
                   rmm::mr::device_memory_resource *mr)
{
    CUDF_EXPECTS(input.num_columns() > 0, "input must have at least one column to determine dtype.");

    auto dtype = input.column(0).type();
    auto output_needs_mask = false;

    for (auto& col : input) {
        CUDF_EXPECTS(dtype == col.type(), "DTYPE mismatch");
        output_needs_mask |= col.nullable();
    }

    auto out = type_dispatcher(dtype, detail::interleave_columns_functor{},
                               input, output_needs_mask,
                               mr);

    return out;
}

} // namespace experimental

} // namespace cudf
