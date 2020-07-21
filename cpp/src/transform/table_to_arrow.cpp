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

#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/transform.hpp>

namespace cudf {
namespace detail {
    template<typename T>
    std::shared_ptr<arrow::Buffer>
    fetch_data_buffer(column_view input_view, arrow::MemoryPool* ar_mr) {
        const int64_t data_size_in_bytes = sizeof(T) * input_view.size();
        std::shared_ptr<arrow::Buffer> data_buffer;
        
        CUDF_EXPECTS(arrow::AllocateBuffer(ar_mr, data_size_in_bytes, &data_buffer).ok(), "Failed to allocate Arrow buffer for data");
        cudaMemcpy(data_buffer->mutable_data(), input_view.data<T>(), data_size_in_bytes, cudaMemcpyDeviceToHost);


        return data_buffer; 
    }

    std::shared_ptr<arrow::Buffer>
    fetch_mask_buffer(column_view input_view, arrow::MemoryPool* ar_mr) {
        const int64_t mask_size_in_bytes = cudf::bitmask_allocation_size_bytes(input_view.size());
        std::shared_ptr<arrow::Buffer> mask_buffer;
        
        if (input_view.has_nulls()) {
            CUDF_EXPECTS(arrow::AllocateBuffer(ar_mr, mask_size_in_bytes, &mask_buffer).ok(), "Failed to allocate Arrow buffer for mask");
            if (input_view.offset() > 0) {
                cudaMemcpy(mask_buffer->mutable_data(), cudf::copy_bitmask(input_view).data(), mask_size_in_bytes, cudaMemcpyDeviceToHost);
            } else {
                cudaMemcpy(mask_buffer->mutable_data(), input_view.null_mask(), mask_size_in_bytes, cudaMemcpyDeviceToHost);
            }

            return mask_buffer;
        }

        return nullptr;
    }


struct dispatch_to_arrow {
    template <typename T>
    std::enable_if_t<is_fixed_width<T>(), std::shared_ptr<arrow::Array>>
    operator()(column_view input_view,
               cudf::type_id id,
               arrow::MemoryPool* ar_mr) {
        
        auto data_buffer = fetch_data_buffer<T>(input_view, ar_mr);
        auto mask_buffer = fetch_mask_buffer(input_view, ar_mr);

        return to_arrow_array(id, static_cast<int64_t>(input_view.size()), data_buffer, mask_buffer, static_cast<int64_t>(input_view.null_count()));
    }


    template <typename T>
    std::enable_if_t<is_compound<T>(), std::shared_ptr<arrow::Array>>
    operator()(column_view input_view,
               cudf::type_id id,
               arrow::MemoryPool* ar_mr) {
        
        auto mask_buffer = fetch_mask_buffer(input_view, ar_mr);
        std::vector<std::shared_ptr<arrow::Array>> child_arrays;

        for (size_type i = 0; i < input_view.num_children(); i++) {
            auto c = input_view.child(i);
            child_arrays.emplace_back(type_dispatcher(c.type(), dispatch_to_arrow{}, c, c.type().id(), ar_mr));
        }

        if (std::is_same<T, cudf::string_view>::value) {
             auto offset_buffer = child_arrays[0]->data()->buffers[1];
             auto data_buffer = child_arrays[1]->data()->buffers[1];
             //return to_arrow_array(id, static_cast<int64_t>(input_view.size()), offset_buffer, data_buffer, mask_buffer, static_cast<int64_t>(input_view.null_count()));
             return std::make_shared<arrow::StringArray>(static_cast<int64_t>(input_view.size()), offset_buffer, data_buffer, mask_buffer, static_cast<int64_t>(input_view.null_count()));
        } else if (std::is_same<T, cudf::dictionary32>::value){
            // Update indices with bit mask buffer
            auto indices = to_arrow_array(type_id::INT32, static_cast<int64_t>(input_view.size()), child_arrays[0]->data()->buffers[1], mask_buffer, static_cast<int64_t>(input_view.null_count()));
            auto dictionary = child_arrays[1];
            return std::make_shared<arrow::DictionaryArray>(arrow::dictionary(indices->type(), dictionary->type()), indices, dictionary);
        } else {
            auto offset_buffer = child_arrays[0]->data()->buffers[1];
            auto data = child_arrays[1];
            return std::make_shared<arrow::ListArray>(arrow::list(data->type()), static_cast<int64_t>(input_view.size()), offset_buffer, data, mask_buffer, static_cast<int64_t>(input_view.null_count()));
        }
    }

    template <typename T>
    std::enable_if_t<(!is_fixed_width<T>()) and (!is_compound<T>()), std::shared_ptr<arrow::Array>>
    operator()(column_view input_view,
               cudf::type_id id,
               arrow::MemoryPool* ar_mr) {
        CUDF_FAIL("Only fixed width and compund types are supported");
    }
};


std::shared_ptr<arrow::Table> to_arrow(table_view input,
                                       std::vector<std::string> const& column_names,
                                       arrow::MemoryPool* ar_mr,
                                       cudaStream_t stream){

    auto num_columns = input.num_columns();

    CUDF_EXPECTS((column_names.size() == 0) or (column_names.size() == num_columns), "column names should be empty or should be equal to number of columns in table");

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::shared_ptr<arrow::Schema> schema;
    bool has_names = column_names.size() > 0;

    size_type itr = 0;
    std::cout<<"Before For"<<std::endl;
    for (auto const& c : input) {
        arrays.emplace_back(type_dispatcher(c.type(), dispatch_to_arrow{}, c, c.type().id(), ar_mr));
        fields.emplace_back(arrow::field(has_names ? column_names[itr] : nullptr, arrays[itr]->type()));
        itr++;
    }

    std::cout<<"After For"<<std::endl;
    schema = arrow::schema(fields);

    return arrow::Table::Make(schema, arrays);
}

} //namespace detail

std::shared_ptr<arrow::Table> to_arrow(table_view input,
                                       std::vector<std::string> const& column_names,
                                       arrow::MemoryPool* ar_mr){
    return detail::to_arrow(input, column_names, ar_mr);
}

} // namespace cudf
