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
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <string>
#include <string_view>

namespace CUDF_EXPORT cudf {

/// @brief CUDF's IR for row-wise columnar operations
namespace row_ir {

enum class target { CUDA = 0 };

struct instance_context;

struct var_info {
  std::string id = {};
  data_type type = data_type{type_id::EMPTY};
  bool nullable  = false;
};

struct untyped_var_info {
  std::string id = {};
};

struct instance_info {
  std::span<var_info const> inputs;
  std::span<untyped_var_info const> outputs;
};

struct target_info {
  target id = target::CUDA;
};

struct instance_context {
 private:
  uint32_t num_tmp_vars_  = 0;
  std::string tmp_prefix_ = "tmp_";

 public:
  instance_context()                                   = default;
  instance_context(instance_context const&)            = default;
  instance_context& operator=(instance_context const&) = default;
  instance_context(instance_context&&)                 = default;
  instance_context& operator=(instance_context&&)      = default;
  ~instance_context()                                  = default;

  std::string make_tmp_id();
};

struct node {
  virtual std::string_view get_id() = 0;

  virtual data_type get_type() = 0;

  virtual bool get_nullable() = 0;

  virtual void instantiate(instance_context& ctx, instance_info const& info) = 0;

  virtual std::string generate_code(instance_context& ctx,
                                    target_info const& info,
                                    instance_info const& instance) = 0;

  virtual ~node() = 0;
};

using opcode = ast::ast_operator;

struct get_input final : node {
 private:
  std::string id_;
  uint32_t input_;
  data_type type_;
  bool nullable_;

 public:
  get_input(uint32_t input);

  get_input(get_input const&) = delete;

  get_input& operator=(get_input const&) = delete;

  get_input(get_input&&) = default;

  get_input& operator=(get_input&&) = default;

  ~get_input() override = default;

  std::string_view get_id() override;

  data_type get_type() override;

  bool get_nullable() override;

  void instantiate(instance_context& ctx, instance_info const& info) override;

  std::string generate_code(instance_context& ctx,
                            target_info const& info,
                            instance_info const& instance) override;
};

struct set_output final : node {
 private:
  std::string id_;
  uint32_t output_;
  std::unique_ptr<node> source_;
  data_type type_;
  bool nullable_;
  std::string output_id_;

 public:
  set_output(uint32_t output, std::unique_ptr<node> source);

  set_output(set_output const&) = delete;

  set_output& operator=(set_output const&) = delete;

  set_output(set_output&&) = default;

  set_output& operator=(set_output&&) = default;

  ~set_output() override = default;

  std::string_view get_id() override;

  data_type get_type() override;

  bool get_nullable() override;

  void instantiate(instance_context& ctx, instance_info const& info) override;

  std::string generate_code(instance_context& ctx,
                            target_info const& info,
                            instance_info const& instance) override;
};

struct operation final : node {
 private:
  std::string id_;
  opcode op_;
  std::vector<std::unique_ptr<node>> operands_;
  data_type type_;
  bool nullable_;

 public:
  operation(opcode op, std::vector<std::unique_ptr<node>> operands);

  operation(operation const&) = delete;

  operation& operator=(operation const&) = delete;

  operation(operation&&) = default;

  operation& operator=(operation&&) = default;

  ~operation() override = default;

  std::string_view get_id() override;

  data_type get_type() override;

  bool get_nullable() override;

  void instantiate(instance_context& ctx, instance_info const& info) override;

  std::string generate_code(instance_context& ctx,
                            target_info const& info,
                            instance_info const& instance) override;
};

struct ast_column_input_ref {
  uint32_t table  = 0;
  uint32_t column = 0;
};

struct ast_named_column_input_ref {
  std::string name = {};
};

struct ast_scalar_input_ref {
  std::reference_wrapper<cudf::scalar const> scalar;
  cudf::ast::generic_scalar_device_view const value;
};

using ast_input_ref =
  std::variant<ast_column_input_ref, ast_named_column_input_ref, ast_scalar_input_ref>;

struct named_table_view {
  table_view table;
  std::map<std::string, uint32_t> column_names;
};

struct transform_args {
  std::vector<cudf::column_view> columns = {};
  std::string transform_udf              = {};
  cudf::data_type output_type            = cudf::data_type{cudf::type_id::EMPTY};
  bool is_ptx                            = false;
  std::optional<void*> user_data         = std::nullopt;
};

struct transform_result {
  transform_args args                      = {};
  std::vector<cudf::column> scalar_columns = {};
};

struct filter_args {
  std::vector<cudf::column_view> columns     = {};
  std::string predicate_udf                  = {};
  bool is_ptx                                = false;
  std::optional<void*> user_data             = std::nullopt;
  std::optional<std::vector<bool>> copy_mask = std::nullopt;
};

struct filter_result {
  filter_args args                         = {};
  std::vector<cudf::column> scalar_columns = {};
};

struct ast_converter {
 private:
  std::vector<ast_input_ref> input_refs_;      // the input refs for the AST
  std::vector<var_info> input_vars_;           // the input variables for the IR
  std::vector<untyped_var_info> output_vars_;  // the output variables for the IR
  std::unique_ptr<set_output> output_ir_;      // the output IR node

 public:
  ast_converter()                                = default;
  ast_converter(ast_converter const&)            = delete;
  ast_converter& operator=(ast_converter const&) = delete;
  ast_converter(ast_converter&&)                 = default;
  ast_converter& operator=(ast_converter&&)      = default;
  ~ast_converter()                               = default;

 private:
  friend class ast::literal;
  friend class ast::column_reference;
  friend class ast::operation;
  friend class ast::column_name_reference;

  std::unique_ptr<row_ir::node> add_ir_node(cudf::ast::literal const& expr);

  std::unique_ptr<row_ir::node> add_ir_node(cudf::ast::column_reference const& expr);

  std::unique_ptr<row_ir::node> add_ir_node(cudf::ast::operation const& expr);

  std::unique_ptr<row_ir::node> add_ir_node(cudf::ast::column_name_reference const& expr);

  [[nodiscard]] std::span<ast_input_ref const> get_input_refs() const;

  // add an AST input/input_reference and return its reference index
  uint32_t add_ast_input_ref(ast_input_ref in);

  void add_input_var(ast_column_input_ref const& in,
                     bool null_aware,
                     named_table_view const& left_table,
                     named_table_view const& right_table);

  void add_input_var(ast_named_column_input_ref const& in,
                     bool null_aware,
                     named_table_view const& left_table,
                     named_table_view const& right_table);

  void add_input_var(ast_scalar_input_ref const& in,
                     bool null_aware,
                     named_table_view const& left_table,
                     named_table_view const& right_table);

  void add_output_var();

 public:
  // [ ] starts from input to output
  // [ ] what will the first one be? how to link?
  // [ ] nullability
  // [ ] scalar or column
  // [ ] run validation pass to check that operators are supported
  // [ ] check that the lhs and rhs types are supported
  // [ ] check that the outputs are correct
  // [ ] separate to external function that will use the IR and use building blocks provided here
  // [ ] remap kernel arguments to ir inputs
  // [ ] generate code from AST
  // [ ] collect outputs
  // steps: generate, instantiate, instantiate inputs and outputs with ids types, null and indices,
  // validate, generate code
  // [ ] how to map for transform and filter

  transform_result as_transform(target target,
                                cudf::ast::expression const& expr,
                                bool null_aware,
                                named_table_view const& left_table,
                                named_table_view const& right_table);

  filter_result as_filter(
    target target,
    cudf::ast::expression const& expr,
    bool null_aware,
    named_table_view const& left_table,
    named_table_view const& right_table,
    std::optional<std::vector<bool>> const& left_table_copy_mask  = std::nullopt,
    std::optional<std::vector<bool>> const& right_table_copy_mask = std::nullopt);
};

}  // namespace row_ir
}  // namespace CUDF_EXPORT cudf
