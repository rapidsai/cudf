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

 #include <cudf/detail/stream_compaction.hpp>
 #include <cudf/table/table_view.hpp>
 #include <cudf/detail/binaryop.hpp>
 #include <cudf/detail/unary.hpp>
 #include <cudf/detail/interop.hpp>
 #include <cudf/detail/copy.hpp>
 #include <cudf/detail/replace.hpp>
 #include <cudf/scalar/scalar_factories.hpp>
 #include <cudf/types.hpp>
 
 #include <rmm/cuda_stream_view.hpp>
 #include <rmm/exec_policy.hpp>
 
namespace cudf {
namespace detail {
std::unique_ptr<table> one_hot_encoding (
                    column_view const& input_column,
                    column_view const& categories,
                    data_type const& output_type,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr) {
    
    std::vector<std::unique_ptr<column>> res;
    res.reserve(static_cast<std::size_t>(categories.size()));
    auto false_scalar = make_fixed_width_scalar(false, stream, mr);

    std::transform(thrust::make_counting_iterator(0), 
                    thrust::make_counting_iterator(categories.size()), 
                    std::back_inserter(res), 
                    [&categories, &input_column, &false_scalar, &output_type, &stream, &mr](auto i){
                        auto element = detail::get_element(categories, i);
                        auto comp = detail::binary_operation(
                            input_column, *element, binary_operator::NULL_EQUALS, data_type{type_id::BOOL8}, stream);
                        if (output_type.id() == type_id::BOOL8) {
                            return replace_nulls(*comp, *false_scalar, stream, mr);
                        } else {
                            auto comp_filled = replace_nulls(*comp, *false_scalar, stream);
                            return detail::cast(*comp_filled, output_type, stream, mr);
                        }
    });

    return std::make_unique<table>(std::move(res));
}

}  // namespace detail
 
 std::unique_ptr<table> one_hot_encoding(
    column_view const& input_column,
    column_view const& categories,
    data_type const& output_type,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
) {
    return detail::one_hot_encoding(input_column, categories, output_type, rmm::cuda_stream_default, mr);
}
}  // namespace cudf
 