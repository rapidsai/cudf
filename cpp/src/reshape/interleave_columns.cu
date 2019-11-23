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

namespace cudf {

namespace experimental {

namespace detail {

template<typename TElement>
struct interleave_columns_selector
{
    table_device_view in;

    TElement __device__ operator()(size_type i)
    {
        auto col_num = i % in.num_columns();
        auto row_num = i / in.num_columns();
        column_device_view in_col = in.column(col_num);
        return in_col.element<TElement>(row_num);
    }
};

struct interleave_columns_functor
{
    template <typename TElement>
    // std::enable_if_t<true, std::unique_ptr<column>>
    std::unique_ptr<column>
    operator()(table_view const& in,
               mask_allocation_policy mask_policy,
               rmm::mr::device_memory_resource *mr,
               cudaStream_t stream)
    {
        auto arch_column = in.column(0);
        auto size = in.num_columns() * in.num_rows();
        auto out = allocate_like(arch_column, size, mask_policy, mr);
        auto device_in = table_device_view::create(in);
        auto device_out = mutable_column_device_view::create(*out);
        auto counting_it = thrust::make_counting_iterator<size_type>(0);

        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          counting_it,
                          counting_it + size,
                          device_out->data<TElement>(),
                          interleave_columns_selector<TElement>{*device_in});

        return out;
    }
};

} // namespace detail

std::unique_ptr<column>
interleave_columns(table_view const& in,
                   rmm::mr::device_memory_resource *mr,
                   cudaStream_t stream)
{
    CUDF_EXPECTS(in.num_columns() > 0, "input must have at least one column to determine dtype.");

    auto arch_column = in.column(0);

    if (in.num_columns() == 0)
    {
        return empty_like(arch_column);
    }
    auto dtype = arch_column.type();
    auto mask_policy = mask_allocation_policy::NEVER;

    for (auto &&col : in) {
        CUDF_EXPECTS(dtype == col.type(), "DTYPE mismatch");
        if (col.nullable()) {
            mask_policy = mask_allocation_policy::ALWAYS;
        }
    }

    auto out = type_dispatcher(dtype, detail::interleave_columns_functor{},
                               in, mask_policy,
                               mr, stream);

    return out;
    // throw new std::runtime_error("");
}

} // namespace experimental

} // namespace cudf
