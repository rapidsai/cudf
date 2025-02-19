
#include "stats_filter_helpers.hpp"

#include "io/parquet/parquet_common.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf::io::parquet::detail {

template <typename ToType, typename FromType>
ToType stats_caster_base::targetType(FromType const value)
{
  if constexpr (cudf::is_timestamp<ToType>()) {
    return static_cast<ToType>(typename ToType::duration{static_cast<typename ToType::rep>(value)});
  } else if constexpr (std::is_same_v<ToType, string_view>) {
    return ToType{nullptr, 0};
  } else {
    return static_cast<ToType>(value);
  }
}

// uses storage type as T
template <typename T, std ::enable_if_t<cudf::is_dictionary<T>() or cudf::is_nested<T>()>*>
T stats_caster_base::convert(uint8_t const* stats_val, size_t stats_size, Type const type)
{
  CUDF_FAIL("unsupported type for stats casting");
}

template <typename T, std::enable_if_t<cudf::is_boolean<T>()>*>
T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
{
  CUDF_EXPECTS(type == BOOLEAN, "Invalid type and stats combination");
  return stats_caster_base::targetType<T>(*reinterpret_cast<bool const*>(stats_val));
}

// integral but not boolean, and fixed_point, and chrono.
template <typename T,
          std ::enable_if_t<(cudf::is_integral<T>() and not cudf::is_boolean<T>()) or
                            cudf::is_fixed_point<T>() or cudf::is_chrono<T>()>*>
T stats_caster_base::convert(uint8_t const* stats_val, size_t stats_size, Type const type)
{
  switch (type) {
    case INT32: return targetType<T>(*reinterpret_cast<int32_t const*>(stats_val));
    case INT64: return targetType<T>(*reinterpret_cast<int64_t const*>(stats_val));
    case INT96:  // Deprecated in parquet specification
      return stats_caster_base::targetType<T>(
        static_cast<__int128_t>(reinterpret_cast<int64_t const*>(stats_val)[0]) << 32 |
        reinterpret_cast<int32_t const*>(stats_val)[2]);
    case BYTE_ARRAY: [[fallthrough]];
    case FIXED_LEN_BYTE_ARRAY:
      if (stats_size == sizeof(T)) {
        // if type size == length of stats_val. then typecast and return.
        if constexpr (cudf::is_chrono<T>()) {
          return stats_caster_base::targetType<T>(
            *reinterpret_cast<typename T::rep const*>(stats_val));
        } else {
          return stats_caster_base::targetType<T>(*reinterpret_cast<T const*>(stats_val));
        }
      }
      // unsupported type
    default: CUDF_FAIL("Invalid type and stats combination");
  }
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>*>
T stats_caster_base::convert(uint8_t const* stats_val, size_t stats_size, Type const type)
{
  switch (type) {
    case FLOAT: return stats_caster_base::targetType<T>(*reinterpret_cast<float const*>(stats_val));
    case DOUBLE:
      return stats_caster_base::targetType<T>(*reinterpret_cast<double const*>(stats_val));
    default: CUDF_FAIL("Invalid type and stats combination");
  }
}

template <typename T, std::enable_if_t<std::is_same_v<T, string_view>>*>
T stats_caster_base::convert(uint8_t const* stats_val, size_t stats_size, Type const type)
{
  switch (type) {
    case BYTE_ARRAY: [[fallthrough]];
    case FIXED_LEN_BYTE_ARRAY:
      return string_view(reinterpret_cast<char const*>(stats_val), stats_size);
    default: CUDF_FAIL("Invalid type and stats combination");
  }
}

stats_expression_converter::stats_expression_converter(ast::expression const& expr,
                                                       size_type const& num_columns)
  : _num_columns{num_columns}
{
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::literal const& expr)
{
  return expr;
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::column_reference const& expr)
{
  CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
               "Statistics AST supports only left table");
  CUDF_EXPECTS(expr.get_column_index() < _num_columns,
               "Column index cannot be more than number of columns in the table");
  return expr;
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::column_name_reference const& expr)
{
  CUDF_FAIL("Column name reference is not supported in statistics AST");
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;
  auto const operands = expr.get_operands();
  auto const op       = expr.get_operator();

  if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
    // First operand should be column reference, second should be literal.
    CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                 "Only binary operations are supported on column reference");
    CUDF_EXPECTS(dynamic_cast<ast::literal const*>(&operands[1].get()) != nullptr,
                 "Second operand of binary operation with column reference must be a literal");
    v->accept(*this);
    // Push literal into the ast::tree
    auto const& literal  = _stats_expr.push(*dynamic_cast<ast::literal const*>(&operands[1].get()));
    auto const col_index = v->get_column_index();
    switch (op) {
      /* transform to stats conditions. op(col, literal)
      col1 == val --> vmin <= val && vmax >= val
      col1 != val --> !(vmin == val && vmax == val)
      col1 >  val --> vmax > val
      col1 <  val --> vmin < val
      col1 >= val --> vmax >= val
      col1 <= val --> vmin <= val
      */
      case ast_operator::EQUAL: {
        auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
        auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
        _stats_expr.push(ast::operation{
          ast::ast_operator::LOGICAL_AND,
          _stats_expr.push(ast::operation{ast_operator::GREATER_EQUAL, vmax, literal}),
          _stats_expr.push(ast::operation{ast_operator::LESS_EQUAL, vmin, literal})});
        break;
      }
      case ast_operator::NOT_EQUAL: {
        auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
        auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
        _stats_expr.push(
          ast::operation{ast_operator::LOGICAL_OR,
                         _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmin, vmax}),
                         _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmax, literal})});
        break;
      }
      case ast_operator::LESS: [[fallthrough]];
      case ast_operator::LESS_EQUAL: {
        auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
        _stats_expr.push(ast::operation{op, vmin, literal});
        break;
      }
      case ast_operator::GREATER: [[fallthrough]];
      case ast_operator::GREATER_EQUAL: {
        auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
        _stats_expr.push(ast::operation{op, vmax, literal});
        break;
      }
      default: CUDF_FAIL("Unsupported operation in Statistics AST");
    };
  } else {
    auto new_operands = visit_operands(operands);
    if (cudf::ast::detail::ast_operator_arity(op) == 2) {
      _stats_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
    } else if (cudf::ast::detail::ast_operator_arity(op) == 1) {
      _stats_expr.push(ast::operation{op, new_operands.front()});
    }
  }
  return _stats_expr.back();
}

std::reference_wrapper<ast::expression const> stats_expression_converter::get_stats_expr() const
{
  return _stats_expr.back();
}

}  // namespace cudf::io::parquet::detail