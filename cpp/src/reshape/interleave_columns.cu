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

#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {
namespace detail {
namespace {

template<typename T>
struct interleave_columns_element_selector
{
    table_device_view input;
    T __device__ operator()(size_type idx)
    {
        auto col_num = idx % input.num_columns();
        auto row_num = idx / input.num_columns();
        auto in_col = input.column(col_num);
        return in_col.element<T>(row_num);
    }
};

struct interleave_columns_bitmask_selector
{
    table_device_view input;
    bool __device__ operator()(size_type mask_idx, size_type bit_idx) {
        auto col_idx = bit_idx % input.num_columns();
        auto row_idx = bit_idx / input.num_columns();
        auto in_col = input.column(col_idx);
        return bit_is_set(in_col.null_mask(), row_idx);
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
        auto out = allocate_like(arch_column, size, mask_policy, mr, stream);
        auto device_in = table_device_view::create(input);
        auto device_out = mutable_column_device_view::create(*out);
        auto counting_it = thrust::make_counting_iterator<size_type>(0);

        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          counting_it,
                          counting_it + size,
                          device_out->data<T>(),
                          interleave_columns_element_selector<T>{*device_in});

        if (out->nullable())
        {
            auto constexpr block_size = 256;
            auto const grid = grid_1d(size, block_size);

            std::vector<bitmask_type*> masks = { device_out->null_mask() };
            rmm::device_vector<bitmask_type*> device_masks{ masks };
            rmm::device_vector<cudf::size_type> device_valid_counts(1, 0);

            using Selector = interleave_columns_bitmask_selector;
            auto selector = Selector{ *device_in };
            auto kernel = select_bitmask_kernel<Selector, block_size>;
            kernel<<<grid.num_blocks, block_size, 0, stream>>>(selector,
                                                               device_masks.data().get(),
                                                               1,
                                                               size,
                                                               device_valid_counts.data().get());

            thrust::host_vector<cudf::size_type> valid_counts(device_valid_counts);

            out->set_null_count(out->size() - valid_counts[0]);
        }

        return out;
    }
};

} // anonymous namespace
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
