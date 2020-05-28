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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/launcher.h>
#include <jit/parser.h>
#include <jit/type.h>
#include "jit/code/code.h"

#include <jit/common_headers.hpp>
#include <timestamps.hpp.jit>
#include <types.hpp.jit>

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
                     cudaStream_t stream)
{
  std::string hash = "prog_transform" + std::to_string(std::hash<std::string>{}(udf));

  std::string cuda_source = code::kernel_header;
  if (is_ptx) {
    cuda_source += cudf::jit::parse_single_function_ptx(
                     udf, "GENERIC_UNARY_OP", cudf::jit::get_type_name(output_type), {0}) +
                   code::kernel;
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
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
  CUDF_EXPECTS(is_fixed_width(input.type()), "Unexpected non-fixed-width type.");

  std::unique_ptr<column> output = make_fixed_width_column(
    output_type, input.size(), copy_bitmask(input), cudf::UNKNOWN_NULL_COUNT, stream, mr);

  if (input.size() == 0) { return output; }

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
  return detail::transform(input, unary_udf, output_type, is_ptx, mr);
}

}  // namespace cudf
