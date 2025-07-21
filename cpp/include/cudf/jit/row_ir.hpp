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

/// @brief CUDF's IR for row-wise columnar operations
namespace row_ir {

enum class target { CUDA = 0 };

struct context;

struct var_info {
  std::string id = {};
  type_id type   = type_id::EMPTY;
  bool nullable  = false;
};

struct instance_info {
  std::span<var_info const> inputs;
  std::span<var_info const> outputs;
};

struct target_info {
  target target          = target::CUDA;
  instance_info instance = {};
};

struct context {
  std::string make_tmp_id();
};

struct node {
  virtual std::string_view get_id() = 0;

  virtual type_id get_type() = 0;

  virtual bool get_null_aware() = 0;

  virtual void instantiate(context& ctx, instance_info const& info) = 0;

  virtual std::string generate_code(context& ctx, target_info const& info) = 0;

  virtual ~node() = 0;
};

using opcode = ast::ast_operator;

struct get_input final : node {
 private:
  std::string id_;
  uint32_t input_;
  type_id type_;
  bool null_aware_;

 public:
  get_input(uint32_t input, bool null_aware);

  get_input(get_input const&) = delete;

  get_input& operator=(get_input const&) = delete;

  get_input(get_input&&) = default;

  get_input& operator=(get_input&&) = default;

  ~get_input() override = default;

  std::string_view get_id() override;

  type_id get_type() override;

  bool get_null_aware() override;

  void instantiate(context& ctx, instance_info const& info) override;

  std::string generate_code(context& ctx, target_info const& info) override;
};

struct set_output final : node {
 private:
  std::string id_;
  uint32_t output_;
  std::unique_ptr<node> source_;
  type_id type_;
  bool null_aware_;
  std::string output_id_;

 public:
  set_output(uint32_t output, std::unique_ptr<node> source);

  set_output(set_output const&) = delete;

  set_output& operator=(set_output const&) = delete;

  set_output(set_output&&) = default;

  set_output& operator=(set_output&&) = default;

  ~set_output() override = default;

  std::string_view get_id() override;

  type_id get_type() override;

  bool get_null_aware() override;

  void instantiate(context& ctx, instance_info const& info) override;

  std::string generate_code(context& ctx, target_info const& info) override;
};

struct operation final : node {
 private:
  std::string id_;
  opcode op_;
  std::vector<std::unique_ptr<node>> operands_;
  type_id type_;
  bool null_aware_;

 public:
  operation(opcode op, std::vector<std::unique_ptr<node>> operands);

  operation(operation const&) = delete;

  operation& operator=(operation const&) = delete;

  operation(operation&&) = default;

  operation& operator=(operation&&) = default;

  ~operation() override = default;

  std::string_view get_id() override;

  type_id get_type() override;

  bool get_null_aware() override;

  void instantiate(context& ctx, instance_info const& info) override;

  std::string generate_code(context& ctx, target_info const& info) override;
};

/*
struct get_input;
struct set_output;

struct input {
  std::string_view id_;
  get_input* dst_;
};

struct output {
  std::string_view id_;
  set_output* src_;
};

   std::vector<input> inputs_;
    std::vector<output> outputs_;
    std::vector<node*> nodes_;
    */

// [ ] starts from input to output
// [ ] what will the first one be? how to link?
// [ ] nullability
// [ ] scalar or column
// [ ] run validation pass to check that operators are supported
// [ ] check that the lhs and rhs types are supported
// [ ] check that the outputs are correct
// [ ] support structs
// [ ] separate to external function that will use the IR and use building blocks provided here

struct code_generator {
  // [ ] remap kernel arguments to ir inputs
  // [ ] generate code from AST
  // [ ] collect outputs
  // steps: generate, instantiate, instantiate inputs and outputs with ids types, null and indices,
  // validate, generate code

  void transform(ast::expression const& output_expression,
                 bool null_aware,
                 table_view const& table,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr);

  void filter(ast::expression const& output_expression,
              bool null_aware,
              table_view const& table,
              rmm::cuda_stream_view stream,
              rmm::device_async_resource_ref mr);
};

}  // namespace row_ir
}  // namespace CUDF_EXPORT cudf
