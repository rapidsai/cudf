/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
namespace detail {

/**
 * @brief cudf IR for operations acting on elements of a row
 */
namespace row_ir {

/**
 * @brief The target for which the IR is generated.
 */
enum class target {
  CUDA = 0  /// < CUDA C++
};

/**
 * @brief The type information of the variable used in the IR.
 */
struct type_info {
  data_type type = data_type{type_id::EMPTY};  ///< The data type of the variable
};

/**
 * @brief The information about the variable used in the IR.
 */
struct var_info {
  std::string id = {};  ///< The variable identifier
  type_info type = {};  ///< The type information of the variable
};

/**
 * @brief The information about the variable used in the IR without type information. The type
 * information is auto-deduced from an IR node.
 */
struct untyped_var_info {
  std::string id = {};  ///< The variable identifier
};

/**
 * @brief The information needed to instantiate the IR nodes
 */
struct instance_info {
  std::span<var_info const> inputs;           ///< The input variables
  std::span<untyped_var_info const> outputs;  ///< The output variables
};

/**
 * @brief The information about the target for which the IR is generated.
 */
struct target_info {
  target id = target::CUDA;  ///< The target identifier
};

/**
 * @brief The context within which the IR is instantiated.
 * This context is used to generate temporary variable identifiers and any state setup needed for
 * the IR instantiation.
 */
struct instance_context {
 private:
  int32_t num_tmp_vars_   = 0;       ///< The number of temporary variables generated
  std::string tmp_prefix_ = "tmp_";  ///< The prefix for temporary variable identifiers

 public:
  instance_context() = default;  ///< Default constructor

  instance_context(instance_context const&) = delete;

  instance_context& operator=(instance_context const&) = delete;

  instance_context(instance_context&&) = default;  ///< Move constructor

  instance_context& operator=(instance_context&&) = default;  ///< Move assignment operator

  ~instance_context() = default;  ///< Destructor

  /**
   * @brief Generate a globally unique temporary variable identifier
   * @return A unique temporary variable identifier
   */
  std::string make_tmp_id();

  void reset();
};

struct node {
  /**
   * @brief Get the identifier of the IR node
   * @return The identifier of the IR node
   */
  virtual std::string_view get_id() = 0;

  /**
   * @brief Get the type info of the IR node
   * @return The type information of the IR node
   */
  virtual type_info get_type() = 0;

  /**
   * @brief Instantiate the IR node with the given context and instance information, setting up any
   * necessary state and preprocessing needed for code generation.
   * @param ctx The context within which the IR is instantiated
   * @param info The instance information
   */
  virtual void instantiate(instance_context& ctx, instance_info const& info) = 0;

  /**
   * @brief Generate the code for the IR node based on the instance context and target information.
   * @param ctx The context within which the IR is instantiated
   * @param info The target information
   * @param instance The instance information
   * @return The generated code for the IR node
   */
  virtual std::string generate_code(instance_context& ctx,
                                    target_info const& info,
                                    instance_info const& instance) = 0;

  virtual ~node() = default;
};

/**
 * @brief The operation code used in the IR nodes.
 */
using opcode = ast::ast_operator;

/**
 * @brief An IR node that retrieves an input variable by its index.
 * This node is used to access input variables in the IR.
 */
struct get_input final : node {
 private:
  std::string id_;  ///< The identifier of the IR node
  int32_t input_;   ///< The index of the input variable
  type_info type_;  ///< The type information of the IR node

 public:
  /**
   * @brief Construct a new get_input IR node
   * @param input The index of the input variable
   */
  get_input(int32_t input);

  get_input(get_input const&) = delete;

  get_input& operator=(get_input const&) = delete;

  get_input(get_input&&) = default;  ///< Move constructor

  get_input& operator=(get_input&&) = default;  ///< Move assignment operator

  ~get_input() override = default;  ///< Destructor

  /**
   * @copydoc node::get_id
   */
  std::string_view get_id() override;

  /**
   * @copydoc node::get_type
   */
  type_info get_type() override;

  /**
   * @copydoc node::instantiate
   */
  void instantiate(instance_context& ctx, instance_info const& info) override;

  /**
   * @copydoc node::generate_code
   */
  std::string generate_code(instance_context& ctx,
                            target_info const& info,
                            instance_info const& instance) override;
};

/**
 * @brief An IR node that sets the output variable to the value of a source IR node.
 */
struct set_output final : node {
 private:
  std::string id_;                ///< The identifier of the IR node
  int32_t output_;                ///< The index of the output variable
  std::unique_ptr<node> source_;  ///< The source IR node from which the value is taken
  type_info type_;                ///< The type information of the IR node
  std::string output_id_;         ///< The identifier of the output variable

 public:
  /**
   * @brief Construct a new set_output IR node
   * @param output The index of the output variable
   * @param source The source IR node from which the value is taken
   */
  set_output(int32_t output, std::unique_ptr<node> source);

  set_output(set_output const&) = delete;

  set_output& operator=(set_output const&) = delete;

  set_output(set_output&&) = default;  ///< Move constructor

  set_output& operator=(set_output&&) = default;  ///< Move assignment operator

  ~set_output() override = default;  ///< Destructor

  /**
   * @copydoc node::get_id
   */
  std::string_view get_id() override;

  /**
   * @copydoc node::get_type
   */
  type_info get_type() override;

  /**
   * @brief Get the source IR node from which the value is taken
   */
  node& get_source();

  /**
   * @copydoc node::instantiate
   */
  void instantiate(instance_context& ctx, instance_info const& info) override;

  /**
   * @copydoc node::generate_code
   */
  std::string generate_code(instance_context& ctx,
                            target_info const& info,
                            instance_info const& instance) override;
};

/**
 * @brief An IR node that represents an operation with zero or more operands.
 */
struct operation final : node {
 private:
  std::string id_;                               ///< The identifier of the IR node
  opcode op_;                                    ///< The operation code
  std::vector<std::unique_ptr<node>> operands_;  ///< The operands of the operation
  type_info type_;                               ///< The type information of the IR node

  operation(opcode op, std::unique_ptr<node>* move_begin, std::unique_ptr<node>* move_end);

 public:
  /**
   * @brief Create a set of operand IR nodes
   */
  template <typename... T>
    requires(std::is_base_of_v<node, T> && ...)
  static std::array<std::unique_ptr<node>, sizeof...(T)> operands(T&&... args)
  {
    return {std::make_unique<T>(std::forward<T>(args))...};
  }

  /**
   * @brief Create a set of operand IR nodes from existing unique pointers
   */
  template <typename... T>
    requires(std::is_base_of_v<node, T> && ...)
  static std::array<std::unique_ptr<node>, sizeof...(T)> operands(std::unique_ptr<T>&&... args)
  {
    return {std::move(args)...};
  }

  /**
   * @brief Construct a new operation IR node
   * @param op The operation code
   * @param operands The operands of the operation
   */
  operation(opcode op, std::vector<std::unique_ptr<node>> operands);

  template <size_t N>
  operation(opcode op, std::array<std::unique_ptr<node>, N> operands)
    : operation{op, operands.data(), operands.data() + N}
  {
  }

  operation(operation const&) = delete;

  operation& operator=(operation const&) = delete;

  operation(operation&&) = default;  ///< Move constructor

  operation& operator=(operation&&) = default;  ///< Move assignment operator

  ~operation() override = default;  ///< Destructor

  /**
   * @copydoc node::get_id
   */
  std::string_view get_id() override;

  /**
   * @copydoc node::get_type
   */
  type_info get_type() override;

  /**
   * @brief Get the operation code of the operation
   * @return The operation code of the operation
   */
  [[nodiscard]] opcode get_opcode() const;

  /** @brief Get the operands of the operation
   * @return A span of unique pointers to the operands of the operation
   */
  [[nodiscard]] std::span<std::unique_ptr<node> const> get_operands() const;

  /**
   * @copydoc node::instantiate
   */
  void instantiate(instance_context& ctx, instance_info const& info) override;

  /**
   * @copydoc node::generate_code
   */
  std::string generate_code(instance_context& ctx,
                            target_info const& info,
                            instance_info const& instance) override;
};

/**
 * @brief A specification of an input column to the AST
 */
struct ast_column_input_spec {
  ast::table_reference table = {};  ///< The table reference (LEFT or RIGHT)
  int32_t column             = 0;   ///< The column index in the referenced table
};

/**
 * @brief A specification of an input scalar to the AST
 */
struct ast_scalar_input_spec {
  std::reference_wrapper<scalar const> ref;  ///< The scalar value
  ast::generic_scalar_device_view view;      ///< The device view of the scalar value
  std::unique_ptr<column> broadcast_column =
    nullptr;  ///< The broadcasted column, a column of size 1
};

/**
 * @brief An input specification for the AST
 */
using ast_input_spec = std::variant<ast_column_input_spec, ast_scalar_input_spec>;

/**
 * @brief The arguments needed to invoke a `cudf::transform`
 */
struct transform_args {
  std::vector<std::unique_ptr<column>> scalar_columns =
    {};  ///< The scalar columns created during the expression conversion
  std::vector<column_view> columns = {};  ///< The input columns to the transform
  std::string udf                  = {};  ///< The user-defined function to apply
  data_type output_type          = data_type{type_id::EMPTY};  ///< The output type of the transform
  bool is_ptx                    = false;           ///< Whether the transform is a PTX kernel
  std::optional<void*> user_data = std::nullopt;    ///< User data to pass to the transform
  null_aware is_null_aware       = null_aware::NO;  ///< Whether the transform is null-aware
};

/**
 * @brief The arguments needed to invoke a `cudf::filter`
 */
struct filter_args {
  std::vector<std::unique_ptr<column>> scalar_columns =
    {};  ///< The scalar columns created during the expression conversion
  std::vector<column_view> predicate_columns = {};  ///< The input columns to the predicate UDF
  std::string predicate_udf = {};  ///< The user-defined function to apply as a predicate
  std::vector<column_view> filter_columns = {};     ///< The input columns to the filter
  bool is_ptx                             = false;  ///< Whether the filter is a PTX device function
  std::optional<void*> user_data          = std::nullopt;    ///< User data to pass to the filter
  null_aware is_null_aware                = null_aware::NO;  ///< Whether the filter is null-aware
};

/**
 * @brief The AST input column arguments used to resolve the column expressions
 */
struct ast_args {
  table_view table = {};  ///< The table view containing the columns
};

/**
 * @brief AST Converter is a class for converting AST expressions to codegen targets, ie. CUDA.
 */
struct ast_converter {
 private:
  std::vector<ast_input_spec> input_specs_;              ///< The input specs for the AST
  std::vector<var_info> input_vars_;                     ///< The input variables for the IR
  std::vector<untyped_var_info> output_vars_;            ///< The output variables for the IR
  std::vector<std::unique_ptr<set_output>> output_irs_;  ///< The output IR nodes
  std::string code_;                                     ///< The generated code for the IR
  rmm::cuda_stream_view
    stream_;  ///< CUDA stream used for device memory operations and kernel launches.
  rmm::device_async_resource_ref
    mr_;  ///< Device memory resource used to allocate the returned table's device memory

 public:
  /**
   * @brief Construct a new AST Converter object
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table's device memory
   */
  ast_converter(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
    : stream_(stream), mr_(mr)
  {
  }

  ast_converter(ast_converter const&)            = delete;
  ast_converter& operator=(ast_converter const&) = delete;

  ast_converter(ast_converter&&) = default;  ///< Move constructor

  ast_converter& operator=(ast_converter&&) = default;  ///< Move assignment operator

  ~ast_converter() = default;  ///< Destructor

 private:
  friend class ast::literal;
  friend class ast::column_reference;
  friend class ast::operation;
  friend class ast::column_name_reference;

  std::unique_ptr<row_ir::node> add_ir_node(ast::literal const& expr);

  std::unique_ptr<row_ir::node> add_ir_node(ast::column_reference const& expr);

  std::unique_ptr<row_ir::node> add_ir_node(ast::operation const& expr);

  [[nodiscard]] std::span<ast_input_spec const> get_input_specs() const;

  /**
   * @brief add an AST input/input_reference and return its reference index
   */
  int32_t add_ast_input(ast_input_spec in);

  void add_input_var(ast_column_input_spec const& in, ast_args const& args);

  void add_input_var(ast_scalar_input_spec const& in, ast_args const& args);

  void add_output_var();

  void generate_code(target target, ast::expression const& expr, ast_args const& args);

 public:
  /**
   * @brief Convert an AST `compute_column` expression to a `cudf::transform`
   * @param target The target for which the IR is generated
   * @param expr The AST expression to convert
   * @param null_aware Whether to use null-aware operators
   * @param args The arguments needed to resolve the AST expression
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table's device memory
   * @return The result of the conversion, containing the transform arguments and scalar columns
   */
  static transform_args compute_column(target target,
                                       ast::expression const& expr,
                                       ast_args const& args,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

  /**
   * @brief Convert an AST `filter` expression to a `cudf::filter`
   * @param target The target for which the IR is generated
   * @param expr The AST expression to convert
   * @param null_aware Whether to use null-aware operators
   * @param args The arguments needed to resolve the AST expression
   * @param filter_table The table to be filtered
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table's device memory
   * @return The result of the conversion, containing the filter arguments and scalar columns
   */
  static filter_args filter(target target,
                            ast::expression const& expr,
                            ast_args const& args,
                            table_view const& filter_table,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);
};

}  // namespace row_ir
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
