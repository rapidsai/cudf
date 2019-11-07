/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <bitmask/legacy/bitmask_ops.hpp> // remove
#include <cudf/utilities/error.hpp>
#include <utilities/cudf_utils.h> // remove
#include <cudf/cudf.h> // remove
#include <bitmask/legacy/legacy_bitmask.hpp> //remove
#include <cudf/legacy/copying.hpp> // remove/replace
#include <cudf/utilities/error.hpp> // wtf duplicate
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/binaryop.hpp>

#include <jit/launcher.h>
#include <jit/type.h>
#include <jit/parser.h>
#include <binaryop/jit/code/code.h>
#include <binaryop/jit/util.hpp>
#include <cudf/datetime.hpp> // replace eventually

#include <types.h.jit>
#include <types.hpp.jit>

namespace cudf {
namespace experimental {

namespace binops {

  // /**---------------------------------------------------------------------------*
  //  * @brief Computes bitwise AND of two input valid masks
  //  *
  //  * This is just a wrapper on apply_bitmask_to_bitmask that can also handle
  //  * cases when one or both of the input masks are nullptr, in which case, it
  //  * copies the mask from the non nullptr input or sets all the output mask to
  //  * valid respectively
  //  *
  //  * @param out_null_coun[out] number of nulls in output
  //  * @param valid_out preallocated output mask
  //  * @param valid_left input mask 1
  //  * @param valid_right input mask 2
  //  * @param num_values number of values in each input mask valid_left and
  //  *valid_right
  //  *---------------------------------------------------------------------------**/
  // void binary_valid_mask_and(cudf::size_type& out_null_count,
  //                            cudf::valid_type* valid_out,
  //                            const cudf::valid_type* valid_left,
  //                            const cudf::valid_type* valid_right,
  //                            cudf::size_type num_values) {
  //   if (num_values == 0) {
  //     out_null_count = 0;
  //     return;
  //   }

  //   if (valid_out == nullptr && valid_left == nullptr && valid_right == nullptr) {
  //     // if both in cols have no mask, then out col is allowed to have no mask
  //     out_null_count = 0;
  //     return;
  //   }

  //   CUDF_EXPECTS((valid_out != nullptr), "Output valid mask pointer is null");

  //   if (valid_left != nullptr && valid_right != nullptr) {
  //     cudaStream_t stream;
  //     CUDA_TRY(cudaStreamCreate(&stream));
  //     auto error = apply_bitmask_to_bitmask(out_null_count, valid_out, valid_left,
  //                                           valid_right, stream, num_values);
  //     CUDA_TRY(cudaStreamSynchronize(stream));
  //     CUDA_TRY(cudaStreamDestroy(stream));
  //     CUDF_EXPECTS(error == GDF_SUCCESS, "Unable to combine bitmasks");
  //   }

  //   cudf::size_type num_bitmask_elements = gdf_num_bitmask_elements(num_values);

  //   if (valid_left == nullptr && valid_right != nullptr) {
  //     CUDA_TRY(cudaMemcpy(valid_out, valid_right, num_bitmask_elements,
  //                         cudaMemcpyDeviceToDevice));
  //   } else if (valid_left != nullptr && valid_right == nullptr) {
  //     CUDA_TRY(cudaMemcpy(valid_out, valid_left, num_bitmask_elements,
  //                         cudaMemcpyDeviceToDevice));
  //   } else if (valid_left == nullptr && valid_right == nullptr) {
  //     CUDA_TRY(cudaMemset(valid_out, 0xff, num_bitmask_elements));
  //   }

  //   cudf::size_type non_nulls;
  //   auto error = gdf_count_nonzero_mask(valid_out, num_values, &non_nulls);
  //   CUDF_EXPECTS(error == GDF_SUCCESS, "Unable to count number of valids");
  //   out_null_count = num_values - non_nulls;
  // }

/**---------------------------------------------------------------------------*
 * @brief Computes output valid mask for op between a column and a scalar
 *
 * @param out_null_coun[out] number of nulls in output
 * @param valid_out preallocated output mask
 * @param valid_col input mask of column
 * @param valid_scalar bool indicating if scalar is valid
 * @param num_values number of values in input mask valid_col
 *---------------------------------------------------------------------------**/
auto scalar_col_valid_mask_and(column_view const& col, scalar const& s) {
  if (col.size() == 0) {
    return rmm::device_buffer{};
  }

  if (not s.is_valid()) {
    return create_null_mask(col.size(), mask_state::ALL_NULL);
  } else if (s.is_valid() && col.nullable()) {
    return copy_bitmask(col);
  } else if (s.is_valid() && not col.nullable()) {
    return rmm::device_buffer{};
}
}

namespace jit {

  const std::string hash = "prog_binop";

  const std::vector<std::string> compiler_flags { "-std=c++14" };
  const std::vector<std::string> headers_name
        { "operation.h" , "traits.h", cudf_types_h, cudf_types_hpp };
  
  std::istream* headers_code(std::string filename, std::iostream& stream) {
      if (filename == "operation.h") {
          stream << code::operation;
          return &stream;
      }
      if (filename == "traits.h") {
          stream << code::traits;
          return &stream;
      }
      return nullptr;
  }

void binary_operation(mutable_column_view& out,
                      scalar const& lhs,
                      column_view const& rhs,
                      binary_operator ope) {
  
  cudf::jit::launcher(
    hash, code::kernel, headers_name, compiler_flags, headers_code
  ).set_kernel_inst(
    "kernel_v_s", // name of the kernel we are launching
    { cudf::jit::get_type_name(out.type()), // list of template arguments
      cudf::jit::get_type_name(rhs.type()),
      cudf::jit::get_type_name(lhs.type()),
      get_operator_name(ope, OperatorType::Reverse) } 
  ).launch(
    out.size(),
    cudf::jit::get_data_ptr(out),
    cudf::jit::get_data_ptr(rhs),
    cudf::jit::get_data_ptr(lhs)
  );

}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator ope) {
  
  cudf::jit::launcher(
    hash, code::kernel, headers_name, compiler_flags, headers_code
  ).set_kernel_inst(
    "kernel_v_s", // name of the kernel we are launching
    { cudf::jit::get_type_name(out.type()), // list of template arguments
      cudf::jit::get_type_name(lhs.type()),
      cudf::jit::get_type_name(rhs.type()),
      get_operator_name(ope, OperatorType::Direct) } 
  ).launch(
    out.size(),
    cudf::jit::get_data_ptr(out),
    cudf::jit::get_data_ptr(lhs),
    cudf::jit::get_data_ptr(rhs)
  );

}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      binary_operator ope) {

  cudf::jit::launcher(
    hash, code::kernel, headers_name, compiler_flags, headers_code
  ).set_kernel_inst(
    "kernel_v_v", // name of the kernel we are launching
    { cudf::jit::get_type_name(out.type()), // list of template arguments
      cudf::jit::get_type_name(lhs.type()),
      cudf::jit::get_type_name(rhs.type()),
      get_operator_name(ope, OperatorType::Direct) } 
  ).launch(
    out.size(),
    cudf::jit::get_data_ptr(out),
    cudf::jit::get_data_ptr(lhs),
    cudf::jit::get_data_ptr(rhs)
  );

}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      const std::string& ptx,
                      const std::string& output_type) {
  
  std::string ptx_hash = 
    hash + "." + std::to_string(std::hash<std::string>{}(ptx + output_type)); 
  std::string cuda_source =
    cudf::jit::parse_single_function_ptx(ptx, "GENERIC_BINARY_OP", output_type) + code::kernel;

  cudf::jit::launcher(
    ptx_hash, cuda_source, headers_name, compiler_flags, headers_code
  ).set_kernel_inst(
    "kernel_v_v", // name of the kernel we are launching
    { cudf::jit::get_type_name(out.type()), // list of template arguments
      cudf::jit::get_type_name(lhs.type()),
      cudf::jit::get_type_name(rhs.type()),
      get_operator_name(GENERIC_BINARY, OperatorType::Direct) } 
  ).launch(
    out.size(),
    cudf::jit::get_data_ptr(out),
    cudf::jit::get_data_ptr(lhs),
    cudf::jit::get_data_ptr(rhs)
  );

}

}  // namespace jit
}  // namespace binops

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator ope,
                                         data_type output_type)
{
  auto new_mask = binops::scalar_col_valid_mask_and(rhs, lhs);
  auto out = make_numeric_column(output_type, rhs.size(), new_mask);

  if (rhs.size() == 0)
    return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ope);
  return out;
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator ope,
                                         data_type output_type)
{
  auto new_mask = binops::scalar_col_valid_mask_and(lhs, rhs);
  auto out = make_numeric_column(output_type, lhs.size(), new_mask);

  if (lhs.size() == 0)
    return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ope);
  return out;  
}

  // std::unique_ptr<column> binary_operation(column_view const& lhs,
  //                                          column_view const& rhs,
  //                                          binary_operator ope,
  //                                          data_type output_type)
  // {
  //   // Check for 0 sized data
  //   if ((lhs.size() == 0) && (rhs.size() == 0)) return;
  //   CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  //   // binops::binary_valid_mask_and(out->null_count, out->valid, lhs->valid,
  //   //                               rhs->valid, rhs->size);

  //   // TODO: This whole shebang should be replaced by jitified header of chrono
  //         gdf_column lhs_tmp{};
  //         gdf_column rhs_tmp{};
  //         // If the columns are GDF_DATE64 or timestamps with different time resolutions,
  //         // cast the least-granular column to the other's resolution before the binop
  //         std::tie(lhs_tmp, rhs_tmp) = cudf::datetime::cast_to_common_resolution(*lhs, *rhs);

  //         if (lhs_tmp.size > 0) { lhs = &lhs_tmp; }
  //         else if (rhs_tmp.size > 0) { rhs = &rhs_tmp; }

  //   binops::jit::binary_operation(out, lhs, rhs, ope);

  //   gdf_column_free(&lhs_tmp);
  //   gdf_column_free(&rhs_tmp);
  // }

  // std::unique_ptr<column> binary_operation(column_view const& lhs,
  //                                          column_view const& rhs,
  //                                          std::string const& ptx,
  //                                          data_type output_type)
  // {  
  //   CUDF_EXPECTS((lhs.size == rhs.size), "Column sizes don't match");

  //   gdf_column output{};

  //   if (lhs.valid != nullptr) {
  //     output = allocate_column(output_type, lhs.size);
  //   } else if (rhs.valid != nullptr) {
  //     output = allocate_column(output_type, rhs.size);
  //   } else {
  //     output = allocate_column(output_type, lhs.size, false); // don't allocate valid for the output
  //   }
    
  //   // Check for 0 sized data
  //   if (lhs.size == 0) return output;

  //   // Check for null data pointer
  //   CUDF_EXPECTS((lhs.data != nullptr) && (rhs.data != nullptr),
  //                "Column data pointers are null");

  //   // Check for datatype
  //   CUDF_EXPECTS( lhs.dtype == GDF_FLOAT32 || 
  //                 lhs.dtype == GDF_FLOAT64 ||
  //                 lhs.dtype == GDF_INT64   ||
  //                 lhs.dtype == GDF_INT32   ||
  //                 lhs.dtype == GDF_INT16,  "Invalid/Unsupported lhs datatype");
  //   CUDF_EXPECTS( rhs.dtype == GDF_FLOAT32 || 
  //                 rhs.dtype == GDF_FLOAT64 ||
  //                 rhs.dtype == GDF_INT64   ||
  //                 rhs.dtype == GDF_INT32   ||
  //                 rhs.dtype == GDF_INT16,  "Invalid/Unsupported rhs datatype");
    
  //   binops::binary_valid_mask_and(output.null_count, output.valid, lhs.valid,
  //                                 rhs.valid, rhs.size);

  //   binops::jit::binary_operation(&output, &lhs, &rhs, ptx,
  //                                 cudf::jit::get_type_name(output_type));
  // }

} // namespace experimental
} // namespace cudf
