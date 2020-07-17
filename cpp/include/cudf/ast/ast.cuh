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
#pragma once

#include <thrust/detail/raw_pointer_cast.h>
#include <algorithm>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <functional>
#include <iterator>
#include <rmm/device_uvector.hpp>
#include <type_traits>
#include "linearizer.hpp"
#include "operators.hpp"

namespace cudf {

namespace ast {

template <typename Element>
__device__ Element resolve_input_data_reference(detail::device_data_reference device_data_reference,
                                                table_device_view const& table,
                                                std::int64_t* thread_intermediate_storage,
                                                cudf::size_type row_index)
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      auto column = table.column(device_data_reference.data_index);
      return column.data<Element>()[row_index];
    }
    case detail::device_data_reference_type::LITERAL: {
      // TODO: Fetch and return literal.
      return static_cast<Element>(0);
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
      return *reinterpret_cast<Element*>(
        &thread_intermediate_storage[device_data_reference.data_index]);
    }
    default: {
      // TODO: Error
      return static_cast<Element>(0);
    }
  }
}

template <typename Element>
__device__ Element* resolve_output_data_reference(
  detail::device_data_reference device_data_reference,
  table_device_view const& table,
  std::int64_t* thread_intermediate_storage,
  cudf::size_type row_index)
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      // TODO: Support output columns?
      return nullptr;
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
      return reinterpret_cast<Element*>(
        &thread_intermediate_storage[device_data_reference.data_index]);
    }
    default: {
      // TODO: Error
      return nullptr;
    }
  }
}

/*
// TODO: Resolve data sources for two tables
template <typename Element>
__device__ Element resolve_data_source(detail::device_data_reference device_data_reference,
                                       table_device_view const& left_table,
                                       table_device_view const& right_table,
                                       cudf::size_type left_row_index,
                                       cudf::size_type right_row_index)
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      if (device_data_reference.table_reference == table_reference::LEFT) {
        auto column = left_table.column(device_data_reference.data_index);
        return column.data<Element>()[left_row_index];
      } else if (device_data_reference.table_reference == table_reference::RIGHT) {
        auto column = right_table.column(device_data_reference.data_index);
        return column.data<Element>()[right_row_index];
      }
    }
    case detail::device_data_reference_type::LITERAL: {
      // TODO: Fetch and return literal.
      return static_cast<Element>(0);
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
      // TODO: Fetch and return intermediate.
      return static_cast<Element>(0);
    }
    default: {
      // TODO: Error
      return static_cast<Element>(0);
    }
  }
}
*/

struct typed_binop_dispatch {
  template <typename Element, std::enable_if_t<cudf::is_numeric<Element>()>* = nullptr>
  __device__ void operator()(ast_operator op,
                             table_device_view const& table,
                             std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index,
                             detail::device_data_reference lhs,
                             detail::device_data_reference rhs,
                             detail::device_data_reference output)
  {
    auto typed_lhs =
      resolve_input_data_reference<Element>(lhs, table, thread_intermediate_storage, row_index);
    auto typed_rhs =
      resolve_input_data_reference<Element>(rhs, table, thread_intermediate_storage, row_index);
    auto typed_output =
      resolve_output_data_reference<Element>(output, table, thread_intermediate_storage, row_index);
    *typed_output = ast_operator_dispatcher_typed(op, do_binop<Element>{}, typed_lhs, typed_rhs);
    /*
    if (row_index == 0) {
      printf("lhs index %i = %f, rhs index %i = %f, output index %i = %f\n",
             lhs.data_index,
             float(typed_lhs),
             rhs.data_index,
             float(typed_rhs),
             output.data_index,
             float(*typed_output));
    }
    */
  }

  template <typename Element, std::enable_if_t<!cudf::is_numeric<Element>()>* = nullptr>
  __device__ void operator()(ast_operator op,
                             table_device_view const& table,
                             std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index,
                             detail::device_data_reference lhs,
                             detail::device_data_reference rhs,
                             detail::device_data_reference output)
  {
    // TODO: How else to make this compile? Need a template to match unsupported types, or prevent
    // the compiler from attempting to compile unsupported types here.
  }
};

__device__ void operate(ast_operator op,
                        table_device_view const& table,
                        std::int64_t* thread_intermediate_storage,
                        cudf::size_type row_index,
                        detail::device_data_reference lhs,
                        detail::device_data_reference rhs,
                        detail::device_data_reference output)
{
  type_dispatcher(lhs.data_type,
                  typed_binop_dispatch{},
                  op,
                  table,
                  thread_intermediate_storage,
                  row_index,
                  lhs,
                  rhs,
                  output);
}

struct output_copy_functor {
  template <typename Element, std::enable_if_t<cudf::is_numeric<Element>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view output_column,
                             table_device_view const& table,
                             std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index,
                             detail::device_data_reference expression_output)
  {
    output_column.element<Element>(row_index) = resolve_input_data_reference<Element>(
      expression_output, table, thread_intermediate_storage, row_index);
  };
  template <typename Element, std::enable_if_t<!cudf::is_numeric<Element>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view output_column,
                             table_device_view const& table,
                             std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index,
                             detail::device_data_reference expression_output){
    // TODO: How else to make this compile? Need a template to match unsupported types, or prevent
    // the compiler from attempting to compile unsupported types here.
  };
};

__device__ void evaluate_row_expression(table_device_view const& table,
                                        detail::device_data_reference* data_references,
                                        // scalar* literals,
                                        ast_operator* operators,
                                        cudf::size_type* operator_source_indices,
                                        cudf::size_type num_operators,
                                        cudf::size_type row_index,
                                        std::int64_t* thread_intermediate_storage,
                                        mutable_column_device_view output)
{
  auto operator_source_index = cudf::size_type(0);
  for (cudf::size_type operator_index(0); operator_index < num_operators; operator_index++) {
    // Execute operator
    auto const& op = operators[operator_index];
    if (is_binary_operator(op)) {
      auto lhs_data_ref    = data_references[operator_source_indices[operator_source_index]];
      auto rhs_data_ref    = data_references[operator_source_indices[operator_source_index + 1]];
      auto output_data_ref = data_references[operator_source_indices[operator_source_index + 2]];
      /*
      if (row_index == 0) {
        printf("Operator id %i is ", operator_index);
        switch (op) {
          case ast_operator::ADD: printf("ADDing "); break;
          case ast_operator::SUB: printf("SUBtracting "); break;
          case ast_operator::MUL: printf("MULtiplying "); break;
          default: break;
        }
        printf("lhs index %i and ", operator_source_indices[operator_source_index]);
        printf("rhs index %i to ", operator_source_indices[operator_source_index + 1]);
        printf("output index %i.\n", operator_source_indices[operator_source_index + 2]);
      }
      */

      operator_source_index += 3;
      operate(op,
              table,
              thread_intermediate_storage,
              row_index,
              lhs_data_ref,
              rhs_data_ref,
              output_data_ref);
    } else {
      // TODO: Support ternary operator
      // Assume operator is unary
      // auto input_data_ref  = data_references[operator_source_indices[operator_source_index]];
      // auto output_data_ref = data_references[operator_source_indices[operator_source_index + 1]];
      operator_source_index += 2;
      // TODO: Unary operations
    }
  }
  // Copy from last data reference to output column
  auto expression_output = data_references[operator_source_indices[operator_source_index - 1]];
  type_dispatcher(expression_output.data_type,
                  output_copy_functor{},
                  output,
                  table,
                  thread_intermediate_storage,
                  row_index,
                  expression_output);
}

__global__ void compute_column_kernel(table_device_view table,
                                      detail::device_data_reference* data_references,
                                      // scalar* literals,
                                      ast_operator* operators,
                                      cudf::size_type* operator_source_indices,
                                      cudf::size_type num_operators,
                                      cudf::size_type num_intermediates,
                                      mutable_column_device_view output)
{
  extern __shared__ std::int64_t intermediate_storage[];
  auto thread_intermediate_storage = &intermediate_storage[threadIdx.x * num_intermediates];
  const cudf::size_type start_idx  = threadIdx.x + blockIdx.x * blockDim.x;
  const cudf::size_type stride     = blockDim.x * gridDim.x;
  const auto num_rows              = table.num_rows();

  for (cudf::size_type row_index = start_idx; row_index < num_rows; row_index += stride) {
    evaluate_row_expression(table,
                            data_references,
                            // literals,
                            operators,
                            operator_source_indices,
                            num_operators,
                            row_index,
                            thread_intermediate_storage,
                            output);
    // output.element<bool>(row_index) = evaluate_expression<Element>(expr, table, row_index);
  }
}

std::unique_ptr<column> compute_column(
  table_view const& table,
  std::reference_wrapper<const expression> expr,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  CUDF_FUNC_RANGE()
  // Linearize the AST
  nvtxRangePush("Linearizing...");
  auto expr_linearizer = linearizer(table);
  expr.get().accept(expr_linearizer);
  auto data_references         = expr_linearizer.get_data_references();
  auto literals                = expr_linearizer.get_literals();
  auto operators               = expr_linearizer.get_operators();
  auto num_operators           = cudf::size_type(operators.size());
  auto operator_source_indices = expr_linearizer.get_operator_source_indices();
  auto expr_data_type          = expr_linearizer.get_root_data_type();
  nvtxRangePop();

  // Create device data
  nvtxRangePush("Creating device data...");
  auto device_data_references =
    rmm::device_uvector<detail::device_data_reference>(data_references.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(device_data_references.data(),
                           data_references.data(),
                           sizeof(detail::device_data_reference) * data_references.size(),
                           cudaMemcpyHostToDevice,
                           stream));
  // TODO: Literals
  // auto device_literals = thrust::device_vector<const scalar>();
  auto device_operators = rmm::device_uvector<cudf::ast::ast_operator>(operators.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(device_operators.data(),
                           operators.data(),
                           sizeof(cudf::ast::ast_operator) * operators.size(),
                           cudaMemcpyHostToDevice,
                           stream));
  auto device_operator_source_indices =
    rmm::device_uvector<cudf::size_type>(operator_source_indices.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(device_operator_source_indices.data(),
                           operator_source_indices.data(),
                           sizeof(cudf::size_type) * operator_source_indices.size(),
                           cudaMemcpyHostToDevice,
                           stream));
  // The stream is synced later when the table_device_view is created.
  // CUDA_TRY(cudaStreamSynchronize(stream));
  nvtxRangePop();

  // Output linearizer info
  /*
  std::cout << "LINEARIZER INFO:" << std::endl;
  std::cout << "Number of data references: " << data_references.size() << std::endl;
  std::cout << "Data references: ";
  for (auto dr : data_references) {
    switch (dr.reference_type) {
      case detail::device_data_reference_type::COLUMN: std::cout << "C"; break;
      case detail::device_data_reference_type::LITERAL: std::cout << "L"; break;
      case detail::device_data_reference_type::INTERMEDIATE: std::cout << "I";
    }
    std::cout << dr.data_index << ", ";
  }
  std::cout << std::endl;
  std::cout << "Number of operators: " << num_operators << std::endl;
  std::cout << "Number of operator source indices: " << operator_source_indices.size() << std::endl;
  std::cout << "Number of literals: " << literals.size() << std::endl;
  std::cout << "Operator source indices: ";
  for (auto v : operator_source_indices) { std::cout << v << ", "; }
  std::cout << std::endl;
  */

  // Create table device view
  nvtxRangePush("Creating table device view...");
  auto table_device   = table_device_view::create(table, stream);
  auto table_num_rows = table.num_rows();
  nvtxRangePop();

  // Prepare output column
  nvtxRangePush("Preparing output column...");
  auto output_column =
    make_fixed_width_column(expr_data_type, table_num_rows, mask_state::UNALLOCATED, stream, mr);
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);
  nvtxRangePop();

  // Configure kernel parameters
  nvtxRangePush("Configuring kernel parameters...");
  auto block_size = 1024;  // TODO: Dynamically determine block size based on shared memory limits
                           // and block size limits
  cudf::detail::grid_1d config(table_num_rows, block_size);
  auto num_intermediates = expr_linearizer.get_intermediate_count();
  auto shmem_size_per_block =
    sizeof(std::int64_t) * num_intermediates * config.num_threads_per_block;
  // std::cout << "Requesting " << shmem_size_per_block << " bytes of shared memory." << std::endl;
  nvtxRangePop();

  // Execute the kernel
  nvtxRangePush("Executing AST kernel...");
  compute_column_kernel<<<config.num_blocks,
                          config.num_threads_per_block,
                          shmem_size_per_block,
                          stream>>>(*table_device,
                                    thrust::raw_pointer_cast(device_data_references.data()),
                                    // device_literals,
                                    thrust::raw_pointer_cast(device_operators.data()),
                                    thrust::raw_pointer_cast(device_operator_source_indices.data()),
                                    num_operators,
                                    num_intermediates,
                                    *mutable_output_device);
  nvtxRangePop();
  return output_column;
}

}  // namespace ast

}  // namespace cudf
