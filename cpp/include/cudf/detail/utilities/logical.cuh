/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/reduce.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Check if a predicate is true for all elements in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns true for all elements.
 *
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 * @tparam BinaryOp **[inferred]** The type of the predicate operator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is true for all elements, false otherwise
 */
template <typename InputIterator, typename BinaryOp>
bool all_of(InputIterator begin, InputIterator end, BinaryOp op, rmm::cuda_stream_view stream)
{
  auto const num_items = std::distance(begin, end);
  auto iter            = thrust::make_transform_iterator(begin, op);
  return reduce(iter, iter + num_items, bool{true}, cuda::std::logical_and<bool>{}, stream);
}

/**
 * @brief Check if a predicate is true for any element in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns true for at least one element.
 *
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 * @tparam BinaryOp **[inferred]** The type of the predicate operator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is true for any element, false otherwise
 */
template <typename InputIterator, typename BinaryOp>
bool any_of(InputIterator begin, InputIterator end, BinaryOp op, rmm::cuda_stream_view stream)
{
  auto const num_items = std::distance(begin, end);
  auto iter            = thrust::make_transform_iterator(begin, op);
  return reduce(iter, iter + num_items, bool{false}, cuda::std::logical_or<bool>{}, stream);
}

/**
 * @brief Check if a predicate is false for all elements in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns false for all elements (i.e., no element satisfies the predicate).
 *
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 * @tparam OutputType **[inferred]** The type of the initial value (unused, for API compatibility)
 * @tparam BinaryOp **[inferred]** The type of the predicate operator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param init Initial value (unused, for API compatibility)
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is false for all elements, false otherwise
 */
template <typename InputIterator, typename OutputType, typename BinaryOp>
bool none_of(InputIterator begin,
             InputIterator end,
             OutputType init,
             BinaryOp op,
             rmm::cuda_stream_view stream)
{
  return not any_of(begin, end, op, stream);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
