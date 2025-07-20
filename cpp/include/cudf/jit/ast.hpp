/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <string>
#include <string_view>

namespace CUDF_EXPORT cudf {

// [ ] how to handle nulls
// [ ] can we use LTO-IR with this? YES!!! void kernel();
// [ ] track nvrtc compilation stages
// [ ] string statistics will become possible, i.e. given upper bounds we can allocate exact memory
// for it

/// @brief CUDF's IR for row-wise columnar operations
namespace ir {

enum class target { CUDA = 0 };

namespace row {

// enum class binary_op{};

// enum class unary_op{};

struct context;

struct input_info {
  type_id type;
  bool nullable;
};

struct instance_info {
  std::span<input_info const> inputs;
};

struct preprocess_info {};

struct target_info {
  target target = ir::target::CUDA;
  // [ ] can we use LTO-IR, i.e. have the udf be a void row_op(void * args);

  // [ ] and have the row_op have any compilation mode possible? PTX, IR, etc. as long as they follow our calling convention

  // [ ] ask ben about LTO-IR

  // [ ] try out LTO-IR

  // [ ] row and column setters and getters overhead if we make them dynamic; also parametrized on scalar and null

  // [ ] cache becomes easy to manage

  // [ ] run lto-ir benchmarks

  // [ ] accessors would either have to be static or dynamic (RDC with context args: types count, inputs, outputs)

  // [ ] support multi-output in transforms and filters
};

struct node {
  virtual std::string_view get_id() = 0;

  virtual void set_id(std::string_view id) = 0;

  virtual data_type get_type() = 0;

  virtual void preprocess(context& ctx, preprocess_info const& info) = 0;

  virtual void instantiate(context& ctx, instance_info const& info) = 0;

  virtual std::string generate_code(context& ctx, target_info const& info) = 0;

  virtual ~node() = 0;
};

using opcode = ast::ast_operator;

struct get_input final : node {
  std::string_view id_;
  uint32_t input_;
  data_type type_;
  bool nullable_;
};

struct set_output final : node {
  std::string_view id_;
  uint32_t output_;
  node* source_;
  data_type type_;
  bool nullable_;
};

template <uint32_t NumArgs>
struct nary_op final : node {
  std::string_view id_;
  opcode op_;
  node* args_[NumArgs];
  data_type type_;
  bool nullable_;
};

using unary_op   = nary_op<1>;
using binary_op  = nary_op<2>;
using ternary_op = nary_op<3>;

struct input {
  get_input* dst_;
  std::string_view id_;
};

struct output {
  set_output* src_;
  std::string_view id_;
};

struct row_op {
  // [ ] starts from input to output
  // [ ] what will the first one be? how to link?
  std::vector<node*> ops_;

  struct system {
    uint32_t num_inputs_      = 0;
    uint32_t num_temporaries_ = 0;
    uint32_t num_outputs_     = 0;
    std::vector<input> inputs_;
    std::vector<output> outputs_;

    void preprocess(input_ref& col)
    {
      auto input_index        = num_inputs_++;
      auto input_id           = std::format("in{}", input_index);
      auto intermediate_index = num_temporaries_++;
      auto intermediate_id    = std::format("tmp{}", intermediate_index);
      col.set_id(intermediate_id);
      inputs_.push_back(input{&col, input_id});
      //   auto cuda = std::format("auto {} = {};", intermediate_id, input_id);
      //   col.set_cuda(cuda);
    }

    void preprocess(set_output& out)
    {
      out.node_->preprocess(*this);
      auto output_index       = num_outputs_++;
      auto output_id          = std::format("out{}", output_index);
      auto intermediate_index = num_temporaries_++;
      auto intermediate_id    = std::format("tmp{}", intermediate_index);
      out.set_id(intermediate_id);
      outputs_.push_back(output{&out, output_id});
      /*  auto cuda                    = std::format(R"***(
          auto {} = {};
          *{} = {};
          )***",
                                intermediate_id,
                                out.node_->get_id(),
                                output_id,
                                intermediate_id);
      */
      //   out.set_cuda(cuda);
    }

    void preprocess(binary_op& node)
    {
      node.lhs_->preprocess(*this);
      node.rhs_->preprocess(*this);
      auto index = num_temporaries_++;
      auto id    = std::format("tmp{}", index);
      node.set_id(id);
      // [ ] we need to make this function type insensitive to nullability
      /*  auto cuda = std::format("auto {} = ast::operator_functor<{}, false>({}, {});",
                                node.get_id(),
                                ast::detail::ast_operator_string(node.op_),
                                node.lhs_->get_id(),
                                node.rhs_->get_id());
       */
      //   node.set_cuda(cuda);
    }

    void preprocess(unary_op& node)
    {
      node.lhs_->preprocess(*this);
      auto index = num_temporaries_++;
      auto id    = std::format("tmp{}", index);
      node.set_id(id);
      /* auto cuda       = std::format("auto {} = ast::operator_functor<{}, false>({});",
                               node.get_id(),
                               ast::detail::ast_operator_string(node.op_),
                               node.lhs_->get_id());
     //   node.set_cuda(cuda);    */
    }
  };

  void visit()
  {
    system sys;
    output_->preprocess(sys);
  }

  // [ ] dispatch using column or scalar view

  void instantiate(std::span<data_type const> types, std::span<bool const> scalar)
  {
    // [ ] nullability
    // [ ] scalar or column
    // [ ] run validation pass to check that operators are supported
    // [ ] check that the lhs and rhs types are supported
    // [ ] check that the outputs are correct
    // [ ] support structs
  }

  // [ ] separate to external function that will use the IR and use building blocks provided here
  void kernel_arguments();
};

}  // namespace row
}  // namespace ir
}  // namespace CUDF_EXPORT cudf
