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

void binary_operation(column_view const& A, 
                      column_view const& B, 
                      std::string const& binary_udf, 
                      data_type output_type, 
                      column_view const& outcol_view,
                      column_view const& outmsk_view,
                      rmm::mr::device_memory_resource* mr)
{
  std::string kernel_name =
  jitify2::reflection::Template("cudf::transformation::jit::binop_kernel")  //
    .instantiate(cudf::jit::get_type_name(outcol_view.type()),  // list of template arguments
                 cudf::jit::get_type_name(A.type()),
                 cudf::jit::get_type_name(B.type()));

  std::string cuda_source = cudf::jit::parse_single_function_ptx(
                     binary_udf, "GENERIC_BINARY_OP", cudf::jit::get_type_name(output_type), {0});

  rmm::cuda_stream_view stream;

  cudf::jit::get_program_cache(*transform_jit_binop_kernel_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"transform/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, stream.value())                                   //
    ->launch(outcol_view.size(),
            cudf::jit::get_data_ptr(outcol_view),
            cudf::jit::get_data_ptr(A),
            cudf::jit::get_data_ptr(B),
            cudf::jit::get_data_ptr(outmsk_view),
            A.null_mask(),
            A.offset(),
            B.null_mask(),
            B.offset()
    );
}
/*
void binary_operation(column_view const& A, 
                      column_view const& B, 
                      std::string const& binary_udf, 
                      data_type output_type, 
                      column_view const& outcol_view,
                      column_view const& outmsk_view,
                      rmm::mr::device_memory_resource* mr)
{

  std::string kernel_name

  std::string hash = "prog_transform" + std::to_string(std::hash<std::string>{}(binary_udf));

  std::cout << binary_udf << std::endl;

  std::string cuda_source = code::kernel_header;
  cuda_source += cudf::jit::parse_single_function_ptx(
                     binary_udf, "GENERIC_BINARY_OP", cudf::jit::get_type_name(output_type), {0});

  cuda_source += code::null_kernel;

  std::cout << cuda_source << std::endl;

  rmm::cuda_stream_view stream;

  // Launch the jitify kernel

  cudf::jit::launcher(hash,
                      cuda_source,
                      header_names,
                      cudf::jit::compiler_flags,
                      headers_code,
                      stream)
    .set_kernel_inst("masked_binary_op_kernel",
                    {
                      cudf::jit::get_type_name(outcol_view.type()), 
                      cudf::jit::get_type_name(A.type()),
                      cudf::jit::get_type_name(B.type()),
                    }
    )
    .launch(outcol_view.size(),
            cudf::jit::get_data_ptr(outcol_view),
            cudf::jit::get_data_ptr(A),
            cudf::jit::get_data_ptr(B),
            cudf::jit::get_data_ptr(outmsk_view),
            A.null_mask(),
            A.offset(),
            B.null_mask(),
            B.offset()
    );

}
*/

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

std::unique_ptr<column> masked_binary_op_inner(column_view const& A, 
                                         column_view const& B, 
                                         std::string const& binary_udf, 
                                         data_type output_type, 
                                         column_view const& outcol_view,
                                         column_view const& outmsk_view,
                                         rmm::mr::device_memory_resource* mr)
{
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  transformation::jit::binary_operation(A, B, binary_udf, output_type, outcol_view, outmsk_view, mr);

  std::unique_ptr<column> output = make_fixed_width_column(
    output_type, A.size(), copy_bitmask(A), cudf::UNKNOWN_NULL_COUNT, stream, mr);


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

std::unique_ptr<column> masked_binary_op(column_view const& A, 
                                         column_view const& B, 
                                         std::string const& binary_udf, 
                                         data_type output_type, 
                                         column_view const& outcol_view,
                                         column_view const& outmsk_view,
                                         rmm::mr::device_memory_resource* mr)
{
  std::cout << "HERE!!" << std::endl;
  return detail::masked_binary_op_inner(A, B, binary_udf, output_type, outcol_view, outmsk_view, mr);
}


}  // namespace cudf
