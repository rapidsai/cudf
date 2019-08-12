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
#include <jit/type.h>
#include <nvstrings/NVCategory.h>
#include <utilities/cudf_utils.h>
#include <bitmask/legacy/bitmask_ops.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <utilities/error_utils.hpp>

#include <utilities/column_utils.hpp>

#include <cudf/legacy/column.hpp>

#include <jit/launcher.h>
#include <jit/type.h>
#include <jit/types_h_jit.h>
#include <jit/parser.h>
#include "jit/code/code.h"

namespace cudf {
namespace transformation {

/**---------------------------------------------------------------------------*
 * @brief Computes output valid mask for op between a column and a scalar
 *
 * @param out_null_coun[out] number of nulls in output
 * @param valid_out preallocated output mask
 * @param valid_col input mask of column
 * @param num_values number of values in input mask valid_col
 *---------------------------------------------------------------------------**/

namespace jit {

void unary_operation(gdf_column& output, const gdf_column& input,
                     const std::string& udf, gdf_dtype output_type, bool is_ptx) {
 
  std::string hash = "prog_tranform." 
    + std::to_string(std::hash<std::string>{}(udf));

  std::string cuda_source;
  if(is_ptx){
    cuda_source = 
      cudf::jit::parse_single_function_ptx(
          udf, "GENERIC_UNARY_OP", 
          cudf::jit::getTypeName(output_type), {0}
          ) + code::kernel;
  }else{  
    cuda_source = 
      cudf::jit::parse_single_function_cuda(
          udf, "GENERIC_UNARY_OP") + code::kernel;
  }
  
  // Launch the jitify kernel
  cudf::jit::launcher(
    hash, cuda_source,
    { cudf_types_h },
    { "-std=c++14" }, nullptr
  ).set_kernel_inst(
    "kernel", // name of the kernel we are launching
    { cudf::jit::getTypeName(output.dtype), // list of template arguments
      cudf::jit::getTypeName(input.dtype) }
  ).launch(
    output.size,
    output.data,
    input.data
  );

}

}  // namespace jit

}  // namespace transformation

gdf_column transform(const gdf_column& input,
                     const std::string& unary_udf,
                     gdf_dtype output_type, bool is_ptx) {
  
  // First create a gdf_column and then call the above function
  gdf_column output = allocate_column(output_type, input.size, input.valid != nullptr);
  
  output.null_count = input.null_count;

  // Check for 0 sized data
  if (input.size == 0){
      return output;
  }

  // Check for null data pointer
  CUDF_EXPECTS((input.data != nullptr), "Input column data pointers are null");

  // Check for datatype
  CUDF_EXPECTS( input.dtype != GDF_STRING && input.dtype != GDF_CATEGORY, 
      "Invalid/Unsupported input datatype" );
  
  if (input.valid != nullptr) {
    gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements(input.size);
    CUDA_TRY(cudaMemcpy(output.valid, input.valid, num_bitmask_elements, cudaMemcpyDeviceToDevice));
  }

  transformation::jit::unary_operation(output, input, unary_udf, output_type, is_ptx);

  return output;
}

}  // namespace cudf
