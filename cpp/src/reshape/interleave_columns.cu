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

#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_device_view.cuh>
#include "cudf/utilities/bit.hpp"

namespace cudf {

namespace experimental {

namespace detail {

template<typename T>
struct interleave_columns_selector
{
    table_device_view input;

    T __device__ operator()(size_type idx)
    {
        auto col_num = idx % input.num_columns();
        auto row_num = idx / input.num_columns();
        column_device_view in_col = input.column(col_num);
        return in_col.element<T>(row_num);
    }
};

struct interleave_columns_validity_selector
{
    table_device_view input;
    size_type out_row_count;

    bitmask_type __device__ operator()(size_type i)
    {
        bitmask_type out = 0b00000000000000000000000000000000;

        const auto num_bits = cudf::detail::size_in_bits<bitmask_type>();

        for (size_type bit = 0; bit < num_bits; bit++) {
            size_type out_row = i * num_bits + bit;
            if (out_row >= out_row_count) {
                break;
            }

            size_type col = out_row % input.num_columns();
            size_type row = out_row / input.num_columns();

            out |= bit_is_set(input.column(col).null_mask(), row) << bit;
        }

        return out;
    }
};

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
               mask_allocation_policy mask_policy,
               rmm::mr::device_memory_resource *mr =
                 rmm::mr::get_default_resource(),
               cudaStream_t stream = 0)
    {
        auto arch_column = input.column(0);
        auto size = input.num_columns() * input.num_rows();
        auto out = allocate_like(arch_column, size, mask_policy, mr);
        auto device_in = table_device_view::create(input);
        auto device_out = mutable_column_device_view::create(*out);
        auto counting_it = thrust::make_counting_iterator<size_type>(0);

        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          counting_it,
                          counting_it + size,
                          device_out->data<T>(),
                          interleave_columns_selector<T>{*device_in});

        if (out->nullable())
        {
            thrust::transform(rmm::exec_policy(stream)->on(stream),
                              counting_it,
                              counting_it + bitmask_allocation_size_bytes(size),
                              device_out->null_mask(),
                              interleave_columns_validity_selector{*device_in, device_out->size()});
        }

        return out;
    }
};

} // namespace detail

std::unique_ptr<column>
interleave_columns(table_view const& input)
{
    CUDF_EXPECTS(input.num_columns() > 0, "input must have at least one column to determine dtype.");

    auto dtype = input.column(0).type();
    auto mask_policy = mask_allocation_policy::NEVER;

    for (auto &&col : input) {
        CUDF_EXPECTS(dtype == col.type(), "DTYPE mismatch");
        if (col.nullable()) {
            mask_policy = mask_allocation_policy::ALWAYS;
        }
    }

    auto out = type_dispatcher(dtype, detail::interleave_columns_functor{},
                               input, mask_policy);

    return out;
}

} // namespace experimental

} // namespace cudf
