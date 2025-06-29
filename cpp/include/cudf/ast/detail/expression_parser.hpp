/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/ast/detail/operators.cuh>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/scan.h>

#include <functional>
#include <numeric>
#include <optional>

namespace CUDF_EXPORT cudf {
namespace ast::detail {

/**
 * @brief Node data reference types.
 *
 * This enum is device-specific. For instance, intermediate data references are generated by the
 * linearization process but cannot be explicitly created by the user.
 */
enum class device_data_reference_type {
  COLUMN,       ///< A value in a table column
  LITERAL,      ///< A literal value
  INTERMEDIATE  ///< An internal temporary value
};

/**
 * @brief A device data reference describes a source of data used by a expression.
 *
 * This is a POD class used to create references describing data type and locations for consumption
 * by the `row_evaluator`.
 */
struct alignas(8) device_data_reference {
  device_data_reference(device_data_reference_type reference_type,
                        cudf::data_type data_type,
                        cudf::size_type data_index,
                        table_reference table_source);

  device_data_reference(device_data_reference_type reference_type,
                        cudf::data_type data_type,
                        cudf::size_type data_index);

  device_data_reference_type const reference_type;  // Source of data
  cudf::data_type const data_type;                  // Type of data
  cudf::size_type const data_index;                 // The column index of a table, index of a
                                                    // literal, or index of an intermediate
  table_reference const table_source;

  bool operator==(device_data_reference const& rhs) const
  {
    return std::tie(data_index, data_type, reference_type, table_source) ==
           std::tie(rhs.data_index, rhs.data_type, rhs.reference_type, rhs.table_source);
  }
};

// Type used for intermediate storage in expression evaluation.
template <bool has_nulls>
using IntermediateDataType = possibly_null_value_t<std::int64_t, has_nulls>;

/**
 * @brief A container of all device data required to evaluate an expression on tables.
 *
 * This struct should never be instantiated directly. It is created by the
 * `expression_parser` on construction, and the resulting member is publicly accessible
 * for passing to kernels for constructing an `expression_evaluator`.
 *
 */
struct expression_device_view {
  device_span<detail::device_data_reference const> data_references;
  device_span<generic_scalar_device_view const> literals;
  device_span<ast_operator const> operators;
  device_span<cudf::size_type const> operator_arities;
  device_span<cudf::size_type const> operator_source_indices;
  cudf::size_type num_intermediates;
};

/**
 * @brief The expression_parser traverses an expression and converts it into a form suitable for
 * execution on the device.
 *
 * This class is part of a "visitor" pattern with the `expression` class.
 *
 * This class does pre-processing work on the host, validating operators and operand data types. It
 * traverses downward from a root expression in a depth-first fashion, capturing information about
 * the expressions and constructing vectors of information that are later used by the device for
 * evaluating the abstract syntax tree as a "linear" list of operators whose input dependencies are
 * resolved into intermediate data storage in shared memory.
 */
class expression_parser {
 public:
  /**
   * @brief Construct a new expression_parser object
   *
   * @param expr The expression to create an evaluable expression_parser for.
   * @param left The left table used for evaluating the abstract syntax tree.
   * @param right The right table used for evaluating the abstract syntax tree.
   */
  expression_parser(expression const& expr,
                    cudf::table_view const& left,
                    std::optional<std::reference_wrapper<cudf::table_view const>> right,
                    bool has_nulls,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
    : _left{left},
      _right{right},
      _expression_count{0},
      _intermediate_counter{},
      _has_nulls(has_nulls)
  {
    expr.accept(*this);
    move_to_device(stream, mr);
  }

  /**
   * @brief Construct a new expression_parser object
   *
   * @param expr The expression to create an evaluable expression_parser for.
   * @param table The table used for evaluating the abstract syntax tree.
   */
  expression_parser(expression const& expr,
                    cudf::table_view const& table,
                    bool has_nulls,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
    : expression_parser(expr, table, {}, has_nulls, stream, mr)
  {
  }

  /**
   * @brief Get the root data type of the abstract syntax tree.
   *
   * @return cudf::data_type
   */
  [[nodiscard]] cudf::data_type output_type() const;

  /**
   * @brief Visit a literal expression.
   *
   * @param expr Literal expression.
   * @return cudf::size_type Index of device data reference for the expression.
   */
  cudf::size_type visit(literal const& expr);

  /**
   * @brief Visit a column reference expression.
   *
   * @param expr Column reference expression.
   * @return cudf::size_type Index of device data reference for the expression.
   */
  cudf::size_type visit(column_reference const& expr);

  /**
   * @brief Visit an expression expression.
   *
   * @param expr Expression expression.
   * @return cudf::size_type Index of device data reference for the expression.
   */
  cudf::size_type visit(operation const& expr);

  /**
   * @brief Visit a column name reference expression.
   *
   * @param expr Column name reference expression.
   * @return cudf::size_type Index of device data reference for the expression.
   */
  cudf::size_type visit(column_name_reference const& expr);
  /**
   * @brief Internal class used to track the utilization of intermediate storage locations.
   *
   * As expressions are being evaluated, they may generate "intermediate" data that is immediately
   * consumed. Rather than manifesting this data in global memory, we can store intermediates of any
   * fixed width type (up to 8 bytes) by placing them in shared memory. This class helps to track
   * the number and indices of intermediate data in shared memory using a give-take model. Locations
   * in shared memory can be "taken" and used for storage, "given back," and then later re-used.
   * This aims to minimize the maximum amount of shared memory needed at any point during the
   * evaluation.
   *
   */
  class intermediate_counter {
   public:
    intermediate_counter() : used_values() {}
    cudf::size_type take();
    void give(cudf::size_type value);
    [[nodiscard]] cudf::size_type get_max_used() const { return max_used; }

   private:
    /**
     * @brief Find the first missing value in a contiguous sequence of integers.
     *
     * From a sorted container of integers, find the first "missing" value.
     * For example, {0, 1, 2, 4, 5} is missing 3, and {1, 2, 3} is missing 0.
     * If there are no missing values, return the size of the container.
     *
     * @return cudf::size_type Smallest value not already in the container.
     */
    [[nodiscard]] cudf::size_type find_first_missing() const;

    std::vector<cudf::size_type> used_values;
    cudf::size_type max_used{0};
  };

  expression_device_view device_expression_data;  ///< The collection of data required to evaluate
                                                  ///< the expression on the device.
  int shmem_per_thread;

 private:
  /**
   * @brief Helper function for adding components (operators, literals, etc) to AST plan
   *
   * @tparam T  The underlying type of the input `std::vector`
   * @param[in]  v  The `std::vector` containing components (operators, literals, etc).
   * @param[in,out]  sizes  The `std::vector` containing the size of each data buffer.
   * @param[in,out]  data_pointers  The `std::vector` containing pointers to each data buffer.
   * @param[in,out]  alignment  The maximum alignment needed for all the extracted size and pointers
   */
  template <typename T>
  void extract_size_and_pointer(std::vector<T> const& v,
                                std::vector<cudf::size_type>& sizes,
                                std::vector<void const*>& data_pointers,
                                cudf::size_type& alignment)
  {
    // sub-type alignment will only work provided the alignment is lesser or equal to
    // alignof(max_align_t) which is the maximum alignment provided by rmm's device buffers
    static_assert(alignof(T) <= alignof(max_align_t));
    auto const data_size = sizeof(T) * v.size();
    sizes.push_back(data_size);
    data_pointers.push_back(v.data());
    alignment = std::max(alignment, static_cast<cudf::size_type>(alignof(T)));
  }

  void move_to_device(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  {
    std::vector<cudf::size_type> sizes;
    std::vector<void const*> data_pointers;
    // use a minimum of 4-byte alignment
    cudf::size_type buffer_alignment = 4;

    extract_size_and_pointer(_data_references, sizes, data_pointers, buffer_alignment);
    extract_size_and_pointer(_literals, sizes, data_pointers, buffer_alignment);
    extract_size_and_pointer(_operators, sizes, data_pointers, buffer_alignment);
    extract_size_and_pointer(_operator_arities, sizes, data_pointers, buffer_alignment);
    extract_size_and_pointer(_operator_source_indices, sizes, data_pointers, buffer_alignment);

    // Create device buffer
    auto buffer_offsets = std::vector<cudf::size_type>(sizes.size());
    thrust::exclusive_scan(sizes.cbegin(),
                           sizes.cend(),
                           buffer_offsets.begin(),
                           cudf::size_type{0},
                           [buffer_alignment](auto a, auto b) {
                             // align each component of the AST program
                             return cudf::util::round_up_safe(a + b, buffer_alignment);
                           });

    auto const buffer_size = buffer_offsets.empty() ? 0 : (buffer_offsets.back() + sizes.back());
    auto host_data_buffer  = std::vector<char>(buffer_size);

    for (unsigned int i = 0; i < data_pointers.size(); ++i) {
      std::memcpy(host_data_buffer.data() + buffer_offsets[i], data_pointers[i], sizes[i]);
    }

    _device_data_buffer = rmm::device_buffer(host_data_buffer.data(), buffer_size, stream, mr);
    stream.synchronize();

    // Create device pointers to components of plan
    auto device_data_buffer_ptr            = static_cast<char const*>(_device_data_buffer.data());
    device_expression_data.data_references = device_span<detail::device_data_reference const>(
      reinterpret_cast<detail::device_data_reference const*>(device_data_buffer_ptr +
                                                             buffer_offsets[0]),
      _data_references.size());
    device_expression_data.literals = device_span<generic_scalar_device_view const>(
      reinterpret_cast<generic_scalar_device_view const*>(device_data_buffer_ptr +
                                                          buffer_offsets[1]),
      _literals.size());
    device_expression_data.operators = device_span<ast_operator const>(
      reinterpret_cast<ast_operator const*>(device_data_buffer_ptr + buffer_offsets[2]),
      _operators.size());
    device_expression_data.operator_arities = device_span<cudf::size_type const>(
      reinterpret_cast<cudf::size_type const*>(device_data_buffer_ptr + buffer_offsets[3]),
      _operators.size());
    device_expression_data.operator_source_indices = device_span<cudf::size_type const>(
      reinterpret_cast<cudf::size_type const*>(device_data_buffer_ptr + buffer_offsets[4]),
      _operator_source_indices.size());
    device_expression_data.num_intermediates = _intermediate_counter.get_max_used();
    shmem_per_thread                         = static_cast<int>(
      (_has_nulls ? sizeof(IntermediateDataType<true>) : sizeof(IntermediateDataType<false>)) *
      device_expression_data.num_intermediates);
  }

  /**
   * @brief Helper function for recursive traversal of expressions.
   *
   * When parsing an expression composed of subexpressions, all subexpressions
   * must be evaluated before an operator can be applied to them. This method
   * performs that recursive traversal (in conjunction with the
   * `expression_parser.visit` and `expression.accept` methods if necessary to
   * descend deeper into an expression tree).
   *
   * @param  operands  The operands to visit.
   *
   * @return The indices of the operands stored in the data references.
   */
  std::vector<cudf::size_type> visit_operands(
    cudf::host_span<std::reference_wrapper<cudf::ast::expression const> const> operands);

  /**
   * @brief Add a data reference to the internal list.
   *
   * @param  data_ref  The data reference to add.
   *
   * @return The index of the added data reference in the internal data references list.
   */
  cudf::size_type add_data_reference(detail::device_data_reference data_ref);

  rmm::device_buffer
    _device_data_buffer;  ///< The device-side data buffer containing the plan information, which is
                          ///< owned by this class and persists until it is destroyed.

  cudf::table_view const& _left;
  std::optional<std::reference_wrapper<cudf::table_view const>> _right;
  cudf::size_type _expression_count;
  intermediate_counter _intermediate_counter;
  bool _has_nulls;
  std::vector<detail::device_data_reference> _data_references;
  std::vector<ast_operator> _operators;
  std::vector<cudf::size_type> _operator_arities;
  std::vector<cudf::size_type> _operator_source_indices;
  std::vector<generic_scalar_device_view> _literals;
};

}  // namespace ast::detail

}  // namespace CUDF_EXPORT cudf
