/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/algorithm/reduce.cuh>

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
 * @tparam TransformOp **[inferred]** The type of the unary transformation operator
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is true for all elements, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool all_of(InputIterator begin, InputIterator end, TransformOp op, rmm::cuda_stream_view stream)
{
  return transform_reduce(begin, end, op, true, cuda::std::logical_and<bool>{}, stream);
}

/**
 * @brief Check if a predicate is true for any element in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns true for at least one element.
 *
 * @tparam TransformOp **[inferred]** The type of the unary transformation operator
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is true for any element, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool any_of(InputIterator begin, InputIterator end, TransformOp op, rmm::cuda_stream_view stream)
{
  return transform_reduce(begin, end, op, false, cuda::std::logical_or<bool>{}, stream);
}

/**
 * @brief Check if a predicate is false for all elements in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns false for all elements (i.e., no element satisfies the predicate).
 *
 * @tparam TransformOp **[inferred]** The type of the predicate operator
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param init Initial value (unused, for API compatibility)
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is false for all elements, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool none_of(InputIterator begin, InputIterator end, TransformOp op, rmm::cuda_stream_view stream)
{
  return not any_of(begin, end, op, stream);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
