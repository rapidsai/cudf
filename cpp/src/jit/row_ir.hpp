/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/io/types.hpp>
#include <cudf/operators/op_attributes.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

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
 * @brief The information about the variable used in the IR.
 */
struct var_info {
  std::string id = {};                         ///< The variable identifier
  data_type type = data_type{type_id::EMPTY};  ///< The data type of the variable
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
struct [[nodiscard]] instance_context {
 private:
  int32_t num_tmp_vars_   = 0;       ///< The number of temporary variables generated
  std::string tmp_prefix_ = "tmp_";  ///< The prefix for temporary variable identifiers
  bool has_nulls_         = false;   ///< If expressions involve null values

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
  [[nodiscard]] std::string make_tmp_id();

  /**
   * @brief Returns true if expressions involve null values
   */
  [[nodiscard]] bool has_nulls() const;

  /**
   * @brief Sets whether expressions involve null values
   * @param has_nulls True if expressions involve null values
   */
  void set_has_nulls(bool has_nulls);
};

struct [[nodiscard]] code_sink {
  void emit(std::string_view code);
};

struct [[nodiscard]] input_reference {
  int32_t index = 0;  ///< The index of the input variable
};

struct [[nodiscard]] output_reference {
  int32_t index = 0;  ///< The index of the output variable
};

struct [[nodiscard]] node {
  std::variant<std::monostate, input_reference, output_reference> reference_ =
    std::monostate{};                           ///< The index of the input/output variable
  opcode op_              = opcode::GET_INPUT;  ///< The operation code
  std::vector<node> args_ = {};                 ///< The arguments of the operation

  data_type type_ = {};  ///< The resolved type information of the IR node

  std::string id_ = {};  ///< The identifier of the IR node

  /**
   * @brief Create a set of argument IR nodes
   */
  template <typename... T>
    requires(std::is_same_v<node, T> && ...)
  static std::array<node, sizeof...(T)> arguments(T&&... args)
  {
    return {std::forward<T>(args)...};
  }

  /**
   * @brief Construct a new operation IR node
   * @param op The operation code
   * @param args The arguments of the operation
   */
  node(opcode op, std::vector<node> args);

  /**
   * @brief Construct a new input reference IR node
   * @param input The index of the input variable
   */
  node(input_reference input);

  /**
   * @brief Construct a new output reference IR node
   * @param output The index of the output variable
   * @param arg The argument node that produces the value to be set to the output variable
   */
  node(output_reference reference, node arg);

  node(node const& other)            = default;  ///< Copy constructor
  node(node&& other)                 = default;  ///< Move constructor
  node& operator=(node const& other) = default;  ///< Copy assignment operator
  node& operator=(node&& other)      = default;  ///< Move assignment operator
  ~node()                            = default;  ///< Destructor

  /**
   * @brief Get the identifier of the IR node
   * @return The identifier of the IR node
   */
  [[nodiscard]] std::string_view get_id() const;

  /**
   * @brief Get the type info of the IR node
   * @return The type information of the IR node
   */
  [[nodiscard]] data_type get_type() const;

  /**
   * @brief Get the operation code of the operation
   * @return The operation code of the operation
   */
  opcode get_opcode() const;

  /** @brief Get the arguments of the operation
   * @return A span of unique pointers to the arguments of the operation
   */
  [[nodiscard]] std::span<node const> get_args() const;

  /**
   * @brief Returns `false` if this node forwards nulls from its inputs to its output.
   * e.g., `ADD` operator is not null-aware because if any of its inputs is null, the output is
   * null. but `NULL_EQUAL` operator is null-aware because it can produce a non-null output even if
   * its inputs are null.
   */
  [[nodiscard]] bool is_null_aware() const;

  /**
   * @brief Returns `true` if this node always produces a valid output even if its inputs are
   * nullable, e.g., `IS_NULL` operator produces a valid boolean output regardless of the
   * nullability of its input.
   */
  [[nodiscard]] bool is_always_valid() const;

  /**
   * @brief Get if the IR node can raise an error during evaluation.
   * @return `true` if the IR node can raise an error during evaluation, `false` otherwise
   */
  [[nodiscard]] bool is_fallible() const;

  /**
   * @brief Instantiate the IR node with the given context and instance information, setting up any
   * necessary state and preprocessing needed for code generation.
   * @param ctx The context within which the IR is instantiated
   * @param info The instance information
   */
  void instantiate(instance_context& ctx, instance_info const& info);

  /**
   * @brief Generate the code for the IR node based on the instance context and target information.
   * @param ctx The context within which the IR is instantiated
   * @param info The target information
   * @param instance The instance information
   * @param sink The code sink to which the generated code is emitted
   */
  void emit_code(instance_context& ctx,
                 target_info const& info,
                 instance_info const& instance,
                 code_sink& sink) const;
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
struct [[nodiscard]] transform_args {
  std::vector<std::unique_ptr<column>> scalar_columns =
    {};  ///< The scalar columns created during the expression conversion
  std::vector<std::variant<column_view, scalar_column_view>> inputs =
    {};                                               ///< The input columns to the transform UDF
  std::string udf       = {};                         ///< The user-defined function to apply
  data_type output_type = data_type{type_id::EMPTY};  ///< The output type of the transform
  cudf::udf_source_type source_type = cudf::udf_source_type::CUDA;  ///< The source type of the UDF
  std::optional<void*> user_data    = std::nullopt;    ///< User data to pass to the transform
  null_aware is_null_aware          = null_aware::NO;  ///< Whether the transform is null-aware
  output_nullability null_policy    = output_nullability::PRESERVE;  ///< Null-transformation policy
  std::optional<size_type> row_size = std::nullopt;  ///< The row size of the transform operation
  std::vector<ast_input_spec> input_specs = {};      ///< The input specs (table ref + column index)
};

/**
 * @brief The arguments needed to invoke a `cudf::filter`
 */
struct [[nodiscard]] filter_args {
  std::vector<std::unique_ptr<column>> scalar_columns =
    {};  ///< The scalar columns created during the expression conversion
  std::vector<std::variant<column_view, scalar_column_view>> inputs =
    {};                                          ///< The input columns to the transform UDF
  std::vector<column_view> filter_columns = {};  ///< The input columns to the filter
  std::string udf                   = {};  ///< The user-defined function to apply as a predicate
  cudf::udf_source_type source_type = cudf::udf_source_type::CUDA;  ///< The source type of the UDF
  std::optional<void*> user_data    = std::nullopt;    ///< User data to pass to the filter
  null_aware is_null_aware          = null_aware::NO;  ///< Whether the filter is null-aware
  output_nullability predicate_nullability =
    output_nullability::PRESERVE;  ///< Null-transformation policy for the predicate output
  std::vector<ast_input_spec> input_specs = {};  ///< The input specs (table ref + column index)
};

/**
 * @brief The AST input column arguments used to resolve the column expressions
 */
struct ast_args {
  table_view table       = {};  ///< The table view containing the columns (single-table case)
  table_view left_table  = {};  ///< The left table for join predicates
  table_view right_table = {};  ///< The right table for join predicates
};

/**
 * @brief AST Converter is a class for converting AST expressions to codegen targets, ie. CUDA.
 */
struct [[nodiscard]] ast_converter {
 private:
  std::vector<ast_input_spec> input_specs_;    ///< The input specs for the AST
  std::vector<var_info> input_vars_;           ///< The input variables for the IR
  std::vector<untyped_var_info> output_vars_;  ///< The output variables for the IR
  std::vector<node> output_irs_;               ///< The output IR nodes
  std::string code_;                           ///< The generated code for the IR
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
    : stream_(std::move(stream)), mr_(std::move(mr))
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

  row_ir::node add_ir_node(ast::literal const& expr);

  row_ir::node add_ir_node(ast::column_reference const& expr);

  row_ir::node add_ir_node(ast::operation const& expr);


  [[nodiscard]] std::span<ast_input_spec const> get_input_specs() const;

  /**
   * @brief add an AST input/input_reference and return its reference index
   */
  [[nodiscard]] int32_t add_ast_input(ast_input_spec in);

  void add_input_var(ast_column_input_spec const& in, ast_args const& args);

  void add_input_var(ast_scalar_input_spec const& in, ast_args const& args);

  void add_output_var();

  [[nodiscard]] std::tuple<null_aware, output_nullability> generate_code(
    target target, ast::expression const& expr, ast_args const& args);

 public:
  /**
   * @brief Convert an AST `compute_column` expression to a `cudf::transform`
   * @param target The target for which the IR is generated
   * @param expr The AST expression to convert
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
