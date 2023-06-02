/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include "cudf_jni_apis.hpp"
#include "jni_compiled_expr.hpp"

namespace {

/** Utility class to read data from the serialized AST buffer generated from Java */
class jni_serialized_ast {
  jbyte const *data_ptr;      // pointer to the current entity to deserialize
  jbyte const *const end_ptr; // pointer to the byte immediately after the AST serialized data

  /** Throws an error if there is insufficient space left to read the specified number of bytes */
  void check_for_eof(std::size_t num_bytes_to_read) {
    if (data_ptr + num_bytes_to_read > end_ptr) {
      throw std::runtime_error("Unexpected end of serialized data");
    }
  }

public:
  jni_serialized_ast(cudf::jni::native_jbyteArray &jni_data)
      : data_ptr(jni_data.begin()), end_ptr(jni_data.end()) {}

  /** Returns true if there is no data remaining to be read */
  bool at_eof() { return data_ptr == end_ptr; }

  /** Read a byte from the serialized AST data buffer */
  jbyte read_byte() {
    check_for_eof(sizeof(jbyte));
    return *data_ptr++;
  }

  /** Read a multi-byte value from the serialized AST data buffer */
  template <typename T> T read() {
    if constexpr (std::is_same_v<T, std::string>) {
      auto const size = read<cudf::size_type>();
      check_for_eof(size);
      auto const result = std::string(reinterpret_cast<char const *>(data_ptr), size);
      data_ptr += size;
      return result;
    } else {
      check_for_eof(sizeof(T));
      // use memcpy since data may be misaligned
      T result;
      memcpy(reinterpret_cast<jbyte *>(&result), data_ptr, sizeof(T));
      data_ptr += sizeof(T);
      return result;
    }
  }

  /** Decode a libcudf data type from the serialized AST data buffer */
  cudf::data_type read_cudf_type() {
    auto const dtype_id = static_cast<cudf::type_id>(read_byte());
    switch (dtype_id) {
      case cudf::type_id::INT8:
      case cudf::type_id::INT16:
      case cudf::type_id::INT32:
      case cudf::type_id::INT64:
      case cudf::type_id::UINT8:
      case cudf::type_id::UINT16:
      case cudf::type_id::UINT32:
      case cudf::type_id::UINT64:
      case cudf::type_id::FLOAT32:
      case cudf::type_id::FLOAT64:
      case cudf::type_id::BOOL8:
      case cudf::type_id::TIMESTAMP_DAYS:
      case cudf::type_id::TIMESTAMP_SECONDS:
      case cudf::type_id::TIMESTAMP_MILLISECONDS:
      case cudf::type_id::TIMESTAMP_MICROSECONDS:
      case cudf::type_id::TIMESTAMP_NANOSECONDS:
      case cudf::type_id::DURATION_DAYS:
      case cudf::type_id::DURATION_SECONDS:
      case cudf::type_id::DURATION_MILLISECONDS:
      case cudf::type_id::DURATION_MICROSECONDS:
      case cudf::type_id::DURATION_NANOSECONDS:
      case cudf::type_id::STRING: {
        return cudf::data_type(dtype_id);
      }
      case cudf::type_id::DECIMAL32:
      case cudf::type_id::DECIMAL64: {
        int32_t const scale = read_byte();
        return cudf::data_type(dtype_id, scale);
      }
      default: throw new std::invalid_argument("unrecognized cudf data type");
    }
  }
};

/**
 * Enumeration of the AST expression types that can appear in the serialized data.
 * NOTE: This must be kept in sync with the NodeType enumeration in AstNode.java!
 */
enum class jni_serialized_expression_type : int8_t {
  VALID_LITERAL = 0,
  NULL_LITERAL = 1,
  COLUMN_REFERENCE = 2,
  UNARY_OPERATION = 3,
  BINARY_OPERATION = 4
};

/**
 * Convert a Java AST serialized byte representing an AST unary operator into the
 * corresponding libcudf AST operator.
 * NOTE: This must be kept in sync with the enumeration in UnaryOperator.java!
 */
cudf::ast::ast_operator jni_to_unary_operator(jbyte jni_op_value) {
  switch (jni_op_value) {
    case 0: return cudf::ast::ast_operator::IDENTITY;
    case 1: return cudf::ast::ast_operator::IS_NULL;
    case 2: return cudf::ast::ast_operator::SIN;
    case 3: return cudf::ast::ast_operator::COS;
    case 4: return cudf::ast::ast_operator::TAN;
    case 5: return cudf::ast::ast_operator::ARCSIN;
    case 6: return cudf::ast::ast_operator::ARCCOS;
    case 7: return cudf::ast::ast_operator::ARCTAN;
    case 8: return cudf::ast::ast_operator::SINH;
    case 9: return cudf::ast::ast_operator::COSH;
    case 10: return cudf::ast::ast_operator::TANH;
    case 11: return cudf::ast::ast_operator::ARCSINH;
    case 12: return cudf::ast::ast_operator::ARCCOSH;
    case 13: return cudf::ast::ast_operator::ARCTANH;
    case 14: return cudf::ast::ast_operator::EXP;
    case 15: return cudf::ast::ast_operator::LOG;
    case 16: return cudf::ast::ast_operator::SQRT;
    case 17: return cudf::ast::ast_operator::CBRT;
    case 18: return cudf::ast::ast_operator::CEIL;
    case 19: return cudf::ast::ast_operator::FLOOR;
    case 20: return cudf::ast::ast_operator::ABS;
    case 21: return cudf::ast::ast_operator::RINT;
    case 22: return cudf::ast::ast_operator::BIT_INVERT;
    case 23: return cudf::ast::ast_operator::NOT;
    case 24: return cudf::ast::ast_operator::CAST_TO_INT64;
    case 25: return cudf::ast::ast_operator::CAST_TO_UINT64;
    case 26: return cudf::ast::ast_operator::CAST_TO_FLOAT64;
    default: throw std::invalid_argument("unexpected JNI AST unary operator value");
  }
}

/**
 * Convert a Java AST serialized byte representing an AST binary operator into the
 * corresponding libcudf AST operator.
 * NOTE: This must be kept in sync with the enumeration in BinaryOperator.java!
 */
cudf::ast::ast_operator jni_to_binary_operator(jbyte jni_op_value) {
  switch (jni_op_value) {
    case 0: return cudf::ast::ast_operator::ADD;
    case 1: return cudf::ast::ast_operator::SUB;
    case 2: return cudf::ast::ast_operator::MUL;
    case 3: return cudf::ast::ast_operator::DIV;
    case 4: return cudf::ast::ast_operator::TRUE_DIV;
    case 5: return cudf::ast::ast_operator::FLOOR_DIV;
    case 6: return cudf::ast::ast_operator::MOD;
    case 7: return cudf::ast::ast_operator::PYMOD;
    case 8: return cudf::ast::ast_operator::POW;
    case 9: return cudf::ast::ast_operator::EQUAL;
    case 10: return cudf::ast::ast_operator::NULL_EQUAL;
    case 11: return cudf::ast::ast_operator::NOT_EQUAL;
    case 12: return cudf::ast::ast_operator::LESS;
    case 13: return cudf::ast::ast_operator::GREATER;
    case 14: return cudf::ast::ast_operator::LESS_EQUAL;
    case 15: return cudf::ast::ast_operator::GREATER_EQUAL;
    case 16: return cudf::ast::ast_operator::BITWISE_AND;
    case 17: return cudf::ast::ast_operator::BITWISE_OR;
    case 18: return cudf::ast::ast_operator::BITWISE_XOR;
    case 19: return cudf::ast::ast_operator::LOGICAL_AND;
    case 20: return cudf::ast::ast_operator::NULL_LOGICAL_AND;
    case 21: return cudf::ast::ast_operator::LOGICAL_OR;
    case 22: return cudf::ast::ast_operator::NULL_LOGICAL_OR;
    default: throw std::invalid_argument("unexpected JNI AST binary operator value");
  }
}

/**
 * Convert a Java AST serialized byte representing an AST table reference into the
 * corresponding libcudf AST table reference.
 * NOTE: This must be kept in sync with the enumeration in TableReference.java!
 */
cudf::ast::table_reference jni_to_table_reference(jbyte jni_value) {
  switch (jni_value) {
    case 0: return cudf::ast::table_reference::LEFT;
    case 1: return cudf::ast::table_reference::RIGHT;
    default: throw std::invalid_argument("unexpected JNI table reference value");
  }
}

/** Functor for type-dispatching the creation of an AST literal */
struct make_literal {
  /** Construct an AST literal from a numeric value */
  template <typename T, std::enable_if_t<cudf::is_numeric<T>()> * = nullptr>
  cudf::ast::literal &operator()(cudf::data_type dtype, bool is_valid,
                                 cudf::jni::ast::compiled_expr &compiled_expr,
                                 jni_serialized_ast &jni_ast) {
    std::unique_ptr<cudf::scalar> scalar_ptr = cudf::make_numeric_scalar(dtype);
    scalar_ptr->set_valid_async(is_valid);
    if (is_valid) {
      T val = jni_ast.read<T>();
      using ScalarType = cudf::scalar_type_t<T>;
      static_cast<ScalarType *>(scalar_ptr.get())->set_value(val);
    }

    auto &numeric_scalar = static_cast<cudf::numeric_scalar<T> &>(*scalar_ptr);
    return compiled_expr.add_literal(std::make_unique<cudf::ast::literal>(numeric_scalar),
                                     std::move(scalar_ptr));
  }

  /** Construct an AST literal from a timestamp value */
  template <typename T, std::enable_if_t<cudf::is_timestamp<T>()> * = nullptr>
  cudf::ast::literal &operator()(cudf::data_type dtype, bool is_valid,
                                 cudf::jni::ast::compiled_expr &compiled_expr,
                                 jni_serialized_ast &jni_ast) {
    std::unique_ptr<cudf::scalar> scalar_ptr = cudf::make_timestamp_scalar(dtype);
    scalar_ptr->set_valid_async(is_valid);
    if (is_valid) {
      T val = jni_ast.read<T>();
      using ScalarType = cudf::scalar_type_t<T>;
      static_cast<ScalarType *>(scalar_ptr.get())->set_value(val);
    }

    auto &timestamp_scalar = static_cast<cudf::timestamp_scalar<T> &>(*scalar_ptr);
    return compiled_expr.add_literal(std::make_unique<cudf::ast::literal>(timestamp_scalar),
                                     std::move(scalar_ptr));
  }

  /** Construct an AST literal from a duration value */
  template <typename T, std::enable_if_t<cudf::is_duration<T>()> * = nullptr>
  cudf::ast::literal &operator()(cudf::data_type dtype, bool is_valid,
                                 cudf::jni::ast::compiled_expr &compiled_expr,
                                 jni_serialized_ast &jni_ast) {
    std::unique_ptr<cudf::scalar> scalar_ptr = cudf::make_duration_scalar(dtype);
    scalar_ptr->set_valid_async(is_valid);
    if (is_valid) {
      T val = jni_ast.read<T>();
      using ScalarType = cudf::scalar_type_t<T>;
      static_cast<ScalarType *>(scalar_ptr.get())->set_value(val);
    }

    auto &duration_scalar = static_cast<cudf::duration_scalar<T> &>(*scalar_ptr);
    return compiled_expr.add_literal(std::make_unique<cudf::ast::literal>(duration_scalar),
                                     std::move(scalar_ptr));
  }

  /** Construct an AST literal from a string value */
  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>> * = nullptr>
  cudf::ast::literal &operator()(cudf::data_type dtype, bool is_valid,
                                 cudf::jni::ast::compiled_expr &compiled_expr,
                                 jni_serialized_ast &jni_ast) {
    std::unique_ptr<cudf::scalar> scalar_ptr = [&]() {
      if (is_valid) {
        std::string val = jni_ast.read<std::string>();
        return std::make_unique<cudf::string_scalar>(val, is_valid);
      } else {
        return std::make_unique<cudf::string_scalar>(rmm::device_buffer{}, is_valid);
      }
    }();

    auto &str_scalar = static_cast<cudf::string_scalar &>(*scalar_ptr);
    return compiled_expr.add_literal(std::make_unique<cudf::ast::literal>(str_scalar),
                                     std::move(scalar_ptr));
  }

  /** Default functor implementation to catch type dispatch errors */
  template <typename T, std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_timestamp<T>() &&
                                         !cudf::is_duration<T>() &&
                                         !std::is_same_v<T, cudf::string_view>> * = nullptr>
  cudf::ast::literal &operator()(cudf::data_type dtype, bool is_valid,
                                 cudf::jni::ast::compiled_expr &compiled_expr,
                                 jni_serialized_ast &jni_ast) {
    throw std::logic_error("Unsupported AST literal type");
  }
};

/** Decode a serialized AST literal */
cudf::ast::literal &compile_literal(bool is_valid, cudf::jni::ast::compiled_expr &compiled_expr,
                                    jni_serialized_ast &jni_ast) {
  auto const dtype = jni_ast.read_cudf_type();
  return cudf::type_dispatcher(dtype, make_literal{}, dtype, is_valid, compiled_expr, jni_ast);
}

/** Decode a serialized AST column reference */
cudf::ast::column_reference &compile_column_reference(cudf::jni::ast::compiled_expr &compiled_expr,
                                                      jni_serialized_ast &jni_ast) {
  auto const table_ref = jni_to_table_reference(jni_ast.read_byte());
  cudf::size_type const column_index = jni_ast.read<int>();
  return compiled_expr.add_column_ref(
      std::make_unique<cudf::ast::column_reference>(column_index, table_ref));
}

// forward declaration
cudf::ast::expression &compile_expression(cudf::jni::ast::compiled_expr &compiled_expr,
                                          jni_serialized_ast &jni_ast);

/** Decode a serialized AST unary expression */
cudf::ast::operation &compile_unary_expression(cudf::jni::ast::compiled_expr &compiled_expr,
                                               jni_serialized_ast &jni_ast) {
  auto const ast_op = jni_to_unary_operator(jni_ast.read_byte());
  cudf::ast::expression &child_expression = compile_expression(compiled_expr, jni_ast);
  return compiled_expr.add_operation(
      std::make_unique<cudf::ast::operation>(ast_op, child_expression));
}

/** Decode a serialized AST binary expression */
cudf::ast::operation &compile_binary_expression(cudf::jni::ast::compiled_expr &compiled_expr,
                                                jni_serialized_ast &jni_ast) {
  auto const ast_op = jni_to_binary_operator(jni_ast.read_byte());
  cudf::ast::expression &left_child = compile_expression(compiled_expr, jni_ast);
  cudf::ast::expression &right_child = compile_expression(compiled_expr, jni_ast);
  return compiled_expr.add_operation(
      std::make_unique<cudf::ast::operation>(ast_op, left_child, right_child));
}

/** Decode a serialized AST expression by reading the expression type and dispatching */
cudf::ast::expression &compile_expression(cudf::jni::ast::compiled_expr &compiled_expr,
                                          jni_serialized_ast &jni_ast) {
  auto const expression_type = static_cast<jni_serialized_expression_type>(jni_ast.read_byte());
  switch (expression_type) {
    case jni_serialized_expression_type::VALID_LITERAL:
      return compile_literal(true, compiled_expr, jni_ast);
    case jni_serialized_expression_type::NULL_LITERAL:
      return compile_literal(false, compiled_expr, jni_ast);
    case jni_serialized_expression_type::COLUMN_REFERENCE:
      return compile_column_reference(compiled_expr, jni_ast);
    case jni_serialized_expression_type::UNARY_OPERATION:
      return compile_unary_expression(compiled_expr, jni_ast);
    case jni_serialized_expression_type::BINARY_OPERATION:
      return compile_binary_expression(compiled_expr, jni_ast);
    default: throw std::invalid_argument("data is not a serialized AST expression");
  }
}

/** Decode a serialized AST into a native libcudf AST and associated resources */
std::unique_ptr<cudf::jni::ast::compiled_expr> compile_serialized_ast(jni_serialized_ast &jni_ast) {
  auto jni_expr_ptr = std::make_unique<cudf::jni::ast::compiled_expr>();
  (void)compile_expression(*jni_expr_ptr, jni_ast);

  if (!jni_ast.at_eof()) {
    throw std::invalid_argument("Extra bytes at end of serialized AST");
  }

  return jni_expr_ptr;
}

} // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ast_CompiledExpression_compile(JNIEnv *env, jclass,
                                                                           jbyteArray jni_data) {
  JNI_NULL_CHECK(env, jni_data, "Serialized AST data is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jbyteArray jbytes(env, jni_data);
    jni_serialized_ast jni_ast(jbytes);
    auto compiled_expr_ptr = compile_serialized_ast(jni_ast);
    jbytes.cancel();
    return reinterpret_cast<jlong>(compiled_expr_ptr.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ast_CompiledExpression_computeColumn(JNIEnv *env,
                                                                                 jclass,
                                                                                 jlong j_ast,
                                                                                 jlong j_table) {
  JNI_NULL_CHECK(env, j_ast, "Compiled AST pointer is null", 0);
  JNI_NULL_CHECK(env, j_table, "Table view pointer is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto compiled_expr_ptr = reinterpret_cast<cudf::jni::ast::compiled_expr const *>(j_ast);
    auto tview_ptr = reinterpret_cast<cudf::table_view const *>(j_table);
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(*tview_ptr, compiled_expr_ptr->get_top_expression());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ast_CompiledExpression_destroy(JNIEnv *env, jclass,
                                                                          jlong jni_handle) {
  try {
    cudf::jni::auto_set_device(env);
    auto ptr = reinterpret_cast<cudf::jni::ast::compiled_expr *>(jni_handle);
    delete ptr;
  }
  CATCH_STD(env, );
}

} // extern "C"
