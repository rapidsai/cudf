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

#include <cudf/cudf.h>
// #include <nvstrings/NVCategory.h>
#include <utilities/cudf_utils.h>
// #include <bitmask/legacy/bitmask_ops.hpp>
// #include <bitmask/legacy/legacy_bitmask.hpp>
#include <bitmask/legacy/bitmask_ops.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cudf/legacy/copying.hpp>
#include <cudf/transform.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/legacy/interop.hpp>
// #include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <utilities/error_utils.hpp>

#include <utilities/column_utils.hpp>

#include <cudf/legacy/column.hpp>

#include <jit/launcher.h>
#include <jit/type.h>
#include <jit/parser.h>
#include "jit/code/code.h"

#include <types.h.jit>
#include <types.hpp.jit>

namespace cudf {
namespace transformation {

/**
 * @brief Computes output valid mask for op between a column and a scalar
 *
 * @param out_null_coun[out] number of nulls in output
 * @param valid_out preallocated output mask
 * @param valid_col input mask of column
 * @param num_values number of values in input mask valid_col
 */

namespace jit {

void unary_operation(mutable_column_view output, column_view input,
                     const std::string& udf, data_type output_type, bool is_ptx) {
 
  std::string hash = "prog_tranform." 
    + std::to_string(std::hash<std::string>{}(udf));

  std::string cuda_source;
  if(is_ptx){
    cuda_source = "\n#include <cudf/types.hpp>\n" +
      cudf::jit::parse_single_function_ptx(
          udf, "GENERIC_UNARY_OP", 
          cudf::jit::get_type_name(output_type), {0}
          ) + code::kernel;
  }else{  
    cuda_source = 
      cudf::jit::parse_single_function_cuda(
          udf, "GENERIC_UNARY_OP") + code::kernel;
  }
  
  // Launch the jitify kernel
  cudf::jit::launcher(
    hash, cuda_source,
    { cudf_types_h, cudf_types_hpp },
    { "-std=c++14" }, nullptr
  ).set_kernel_inst(
    "kernel", // name of the kernel we are launching
    { cudf::jit::get_type_name(output.type()), // list of template arguments
      cudf::jit::get_type_name(input.type()) }
  ).launch(
    output.size(),
    // TODO: replace with type dispatched (in/out)put.data<T>()
    // Sad that we still need to use type_dispatcher in Jitified functionality
    output.head<void>(),
    input.head<void>()
  );

}

}  // namespace jit

}  // namespace transformation

std::unique_ptr<column> transform(column_view input,
                                  const std::string &unary_udf,
                                  data_type output_type, bool is_ptx)
{
  std::unique_ptr<column> output =
    make_numeric_column(output_type, input.size(), 
      (input.nullable()) ? cudf::UNINITIALIZED : cudf::UNALLOCATED);

  if (input.size() == 0) {
    return output;
  }

  CUDF_EXPECTS(input.type().id() != cudf::type_id::STRING &&
               input.type().id() != cudf::type_id::CATEGORY,
    "Invalid/Unsupported input datatype");

  mutable_column_view output_view = *output;

  if (input.nullable()) {
    size_t num_bitmask_elements = bitmask_allocation_size_bytes(input.size());
    CUDA_TRY(cudaMemcpy(output_view.null_mask(), input.null_mask(),
                        num_bitmask_elements, cudaMemcpyDeviceToDevice));
  }

  // transform
  transformation::jit::unary_operation(output_view, input, unary_udf, output_type, is_ptx);

  return output;
}

}  // namespace cudf
