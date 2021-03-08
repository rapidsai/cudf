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

#include "jit/code/code.h"

#include <jit/launcher.h>
#include <jit/parser.h>
#include <jit/type.h>
#include <jit/common_headers.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/timestamps.hpp.jit>
#include <jit/types.hpp.jit>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace transformation {
//! Jit functions
namespace jit {

const std::vector<std::string> header_names{cudf_types_hpp, cudf_wrappers_timestamps_hpp};

std::istream* headers_code(std::string filename, std::iostream& stream)
{
  auto it = cudf::jit::stringified_headers.find(filename);
  if (it != cudf::jit::stringified_headers.end()) {
    return cudf::jit::send_stringified_header(stream, it->second);
  }
  return nullptr;
}

void unary_operation(mutable_column_view output,
                     column_view input,
                     const std::string& udf,
                     data_type output_type,
                     bool is_ptx,
                     rmm::cuda_stream_view stream)
{
  std::string hash = "prog_transform" + std::to_string(std::hash<std::string>{}(udf));

  std::cout << "The program's hash is:" << std::endl;
  std::cout << hash << std::endl;

  std::cout << "the actual udf string is: " << std::endl;
  std::cout << udf << std::endl;


  std::cout << "cuda_source is:" << std::endl;
  std::string cuda_source = code::kernel_header;
  std::cout << cuda_source << std::endl;

  if (is_ptx) {
    cuda_source += cudf::jit::parse_single_function_ptx(
                     udf, "GENERIC_UNARY_OP", cudf::jit::get_type_name(output_type), {0}) +
                   code::kernel;
    std::cout << "cuda_source after is_ptx condition: " << std::endl;
    std::cout << cuda_source << std::endl;
  } else {
    cuda_source += cudf::jit::parse_single_function_cuda(udf, "GENERIC_UNARY_OP") + code::kernel;
  }

  // Launch the jitify kernel
  cudf::jit::launcher(hash,
                      cuda_source,
                      header_names,
                      cudf::jit::compiler_flags,
                      headers_code,
                      stream)
    .set_kernel_inst("kernel",  // name of the kernel we are launching
                     {cudf::jit::get_type_name(output.type()),  // list of template arguments
                      cudf::jit::get_type_name(input.type())})
    .launch(output.size(), cudf::jit::get_data_ptr(output), cudf::jit::get_data_ptr(input));
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
                                         rmm::mr::device_memory_resource* mr)
{
  std::cout << "ehllo " << std::endl;
  std::cout << binary_udf << std::endl;

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;



  std::unique_ptr<column> output = make_fixed_width_column(
    output_type, A.size(), copy_bitmask(A), cudf::UNKNOWN_NULL_COUNT, stream, mr);

  auto null_mask = cudf::create_null_mask(A.size(), mask_state::ALL_VALID, mr);

  std::unique_ptr<column> output_mask = make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, A.size(), null_mask, cudf::UNKNOWN_NULL_COUNT, stream, mr);

  

  return output;
}



}  // namespace cudf
