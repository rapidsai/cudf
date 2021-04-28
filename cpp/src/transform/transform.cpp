/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp>
#include <jit_preprocessed_files/transform/jit/binop_kernel.cu.jit.hpp>
#include <jit_preprocessed_files/transform/jit/baked_udf_requirements.cu.jit.hpp>


#include <jit/cache.hpp>
#include <jit/parser.hpp>
#include <jit/type.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace transformation {
namespace jit {

void unary_operation(mutable_column_view output,
                     column_view input,
                     const std::string& udf,
                     data_type output_type,
                     bool is_ptx,
                     rmm::cuda_stream_view stream)
{
  std::string kernel_name =
    jitify2::reflection::Template("cudf::transformation::jit::kernel")  //
      .instantiate(cudf::jit::get_type_name(output.type()),  // list of template arguments
                   cudf::jit::get_type_name(input.type()));

  std::string cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(udf,  //
                                                  "GENERIC_UNARY_OP",
                                                  cudf::jit::get_type_name(output_type),
                                                  {0})
           : cudf::jit::parse_single_function_cuda(udf,  //
                                                   "GENERIC_UNARY_OP");

  cudf::jit::get_program_cache(*transform_jit_kernel_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"transform/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, stream.value())                                   //
    ->launch(output.size(),                                                                 //
             cudf::jit::get_data_ptr(output),
             cudf::jit::get_data_ptr(input));
}

void binary_operation(table_view data_view,
                      std::string const& binary_udf, 
                      data_type output_type, 
                      column_view const& outcol_view,
                      column_view const& outmsk_view,
                      rmm::mr::device_memory_resource* mr)
{

  std::vector<std::string> template_types(
    // A ptr, mask ptr, and offset for each column
    // plus one for the type of the output column
    (data_view.num_columns() * 3) + 1
  );
  template_types[0] = cudf::jit::get_type_name(outcol_view.type());
  for (int i = 0; i < data_view.num_columns(); i++) {
    int offset = (i * 3) + 1;
    template_types[offset] = cudf::jit::get_type_name(data_view.column(i).type()) + "*";
    template_types[offset + 1] = "uint32_t*"; 
    template_types[offset + 2] = "int64_t";
  }


  column_view A = data_view.column(0);
  column_view B = data_view.column(1);

  std::string generic_kernel_name = 
  jitify2::reflection::Template("cudf::transformation::jit::generic_udf_kernel")
    .instantiate(template_types);

  std::string generic_cuda_source = cudf::jit::parse_single_function_ptx(
                     binary_udf, "GENERIC_OP", cudf::jit::get_type_name(output_type), {0});
                     
  rmm::cuda_stream_view generic_stream;
  cudf::jit::get_program_cache(*transform_jit_binop_kernel_cu_jit)
    .get_kernel(
      generic_kernel_name, {}, {{"transform/jit/operation-udf.hpp", generic_cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, generic_stream.value())                                   //
    ->launch(outcol_view.size(),
             cudf::jit::get_data_ptr(outcol_view),
             cudf::jit::get_data_ptr(outmsk_view), 
             cudf::jit::get_data_ptr(A),
             A.null_mask(), // cudf::bitmask_type * 
             A.offset(),
             cudf::jit::get_data_ptr(B),
             B.null_mask(),
             B.offset());

}

void generalized_operation(table_view const& data_view,
                           std::string const& udf,
                           data_type output_type,
                           column_view const& outcol_view,
                           column_view const& outmsk_view,
                           rmm::mr::device_memory_resource* mr)
{
  rmm::cuda_stream_view stream;
  //std::string cuda_source = cudf::jit::parse_single_function_ptx(
  //                   udf, "GENERIC_OP", cudf::jit::get_type_name(output_type), {0});
  /*
  size_t num_cols = data_view.num_columns();
  std::vector<std::string> input_types(num_cols);
  std::vector<void*> args(num_cols);


  column_view this_view;
  for (size_t i = 0; i < num_cols; i++) {
    this_view = data_view.column(i);
    input_types[i] = cudf::jit::get_type_name(this_view.type());
  }
  */

  std::string kernel_name =
    jitify2::reflection::Template("genop_kernel")  //
      .instantiate(cudf::jit::get_type_name(outcol_view.type()));

  cudf::jit::get_program_cache(*transform_jit_baked_udf_requirements_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"transform/jit/operation-udf.hpp", udf}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, stream.value())                                   //
    ->launch(outcol_view.size(),
             static_cast<cudf::size_type>(7),                                                                 //
             cudf::jit::get_data_ptr(outcol_view));

}

}  // namespace jit
}  // namespace transformation

namespace detail {
std::unique_ptr<column> transform(column_view const& input,
                                  std::string const& unary_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_fixed_width(input.type()), "Unexpected non-fixed-width type.");

  std::unique_ptr<column> output = make_fixed_width_column(
    output_type, input.size(), copy_bitmask(input), cudf::UNKNOWN_NULL_COUNT, stream, mr);

  if (input.is_empty()) { return output; }

  mutable_column_view output_view = *output;

  // transform
  transformation::jit::unary_operation(output_view, input, unary_udf, output_type, is_ptx, stream);

  return output;
}

std::unique_ptr<column> masked_binary_op_inner(table_view data_view, 
                                               std::string const& binary_udf, 
                                               data_type output_type, 
                                               column_view const& outcol_view,
                                               column_view const& outmsk_view,
                                               rmm::mr::device_memory_resource* mr)
{
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  transformation::jit::binary_operation(data_view, binary_udf, output_type, outcol_view, outmsk_view, mr);

  std::unique_ptr<column> output;


  return output;
}

std::unique_ptr<column> generalized_masked_op_inner(
  table_view const& data_view,
  std::string const& udf,
  data_type output_type,
  column_view const& outcol_view,
  column_view const& outmsk_view,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  transformation::jit::generalized_operation(data_view, udf, output_type, outcol_view, outmsk_view, mr);

  std::unique_ptr<column> output;

  return output;

}



}  // namespace detail

std::unique_ptr<column> transform(column_view const& input,
                                  std::string const& unary_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::transform(input, unary_udf, output_type, is_ptx, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> masked_binary_op(table_view data_view,
                                         std::string const& binary_udf, 
                                         data_type output_type, 
                                         column_view const& outcol_view,
                                         column_view const& outmsk_view,
                                         rmm::mr::device_memory_resource* mr)
{
  return detail::masked_binary_op_inner(data_view, binary_udf, output_type, outcol_view, outmsk_view, mr);
}

std::unique_ptr<column> generalized_masked_op(
  table_view const& data_view,
  std::string const& udf,
  data_type output_type,
  column_view const& outcol_view,
  column_view const& outmsk_view,
  rmm::mr::device_memory_resource* mr)
{
  return detail::generalized_masked_op_inner(data_view, udf, output_type, outcol_view, outmsk_view, mr);
}


}  // namespace cudf
