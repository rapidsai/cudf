/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <algorithm>

namespace CUDF_EXPORT cudf {

/**
 * @brief Compare the type IDs of two `column_view`s
 *
 * This function returns true if the type of `lhs` equals that of `rhs`.
 * - For fixed point types, the scale is ignored.
 *
 * @param lhs The first `column_view` to compare
 * @param rhs The second `column_view` to compare
 * @return true if column types match
 */
bool column_types_equivalent(column_view const& lhs, column_view const& rhs);

/**
 * @brief Compares the type of two `column_view`s
 *
 * This function returns true if the type of `lhs` equals that of `rhs`.
 * - For fixed point types, the scale is compared.
 * - For dictionary types, the type of the keys are compared if both are
 *   non-empty columns.
 * - For lists types, the type of child columns are compared recursively.
 * - For struct types, the type of each field are compared in order.
 * - For all other types, the `id` of `data_type` is compared.
 *
 * @param lhs The first `column_view` to compare
 * @param rhs The second `column_view` to compare
 * @return true if types match
 */
bool have_same_types(column_view const& lhs, column_view const& rhs);

/**
 * @brief Compare the types of a `column_view` and a `scalar`
 *
 * This function returns true if the type of `lhs` equals that of `rhs`.
 * - For fixed point types, the scale is compared.
 * - For dictionary column types, the type of the keys is compared to the
 *   scalar type.
 * - For lists types, the types of child columns are compared recursively.
 * - For struct types, the types of each field are compared in order.
 * - For all other types, the `id` of `data_type` is compared.
 *
 * @param lhs The `column_view` to compare
 * @param rhs The `scalar` to compare
 * @return true if types match
 */
bool have_same_types(column_view const& lhs, scalar const& rhs);

/**
 * @brief Compare the types of a `scalar` and a `column_view`
 *
 * This function returns true if the type of `lhs` equals that of `rhs`.
 * - For fixed point types, the scale is compared.
 * - For dictionary column types, the type of the keys is compared to the
 *   scalar type.
 * - For lists types, the types of child columns are compared recursively.
 * - For struct types, the types of each field are compared in order.
 * - For all other types, the `id` of `data_type` is compared.
 *
 * @param lhs The `scalar` to compare
 * @param rhs The `column_view` to compare
 * @return true if types match
 */
bool have_same_types(scalar const& lhs, column_view const& rhs);

/**
 * @brief Compare the types of two `scalar`s
 *
 * This function returns true if the type of `lhs` equals that of `rhs`.
 * - For fixed point types, the scale is compared.
 * - For lists types, the types of child columns are compared recursively.
 * - For struct types, the types of each field are compared in order.
 * - For all other types, the `id` of `data_type` is compared.
 *
 * @param lhs The first `scalar` to compare
 * @param rhs The second `scalar` to compare
 * @return true if types match
 */
bool have_same_types(scalar const& lhs, scalar const& rhs);

/**
 * @brief Checks if two `table_view`s have columns of same types
 *
 * @param lhs left-side table_view operand
 * @param rhs right-side table_view operand
 * @return boolean comparison result
 */
bool have_same_types(table_view const& lhs, table_view const& rhs);

/**
 * @brief Compare the types of a range of `column_view` or `scalar` objects
 *
 * This function returns true if all objects in the range have the same type, in the sense of
 * cudf::have_same_types.
 *
 * @tparam ForwardIt Forward iterator
 * @param first The first iterator
 * @param last The last iterator
 * @return true if all types match
 */
template <typename ForwardIt>
inline bool all_have_same_types(ForwardIt first, ForwardIt last)
{
  return first == last || std::all_of(std::next(first), last, [want = *first](auto const& c) {
           return cudf::have_same_types(want, c);
         });
}

}  // namespace CUDF_EXPORT cudf
