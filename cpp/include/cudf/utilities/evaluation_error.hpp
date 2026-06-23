/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/errc.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {

/**
 * @brief Exception type thrown when evaluating an operator function results in an error (e.g.
 * overflow, division by zero, etc.)
 *
 */
struct evaluation_error : public std::exception {
  /**
   * @brief Construct a new evaluation error object.
   *
   * @param error The maximum error code that occurred during evaluation
   * @param row_errors An optional device vector containing per-row error codes (if error_policy is
   * set to PER_ROW)
   * @param message An error message describing the evaluation error
   */
  evaluation_error(errc error,
                   std::optional<rmm::device_uvector<errc>> row_errors,
                   std::string message)
    : max_error_(error), row_errors_(std::move(row_errors)), message_(std::move(message))
  {
  }

  /**
   * @brief Get the error message.
   * @return A C-string describing the error
   */
  [[nodiscard]] char const* what() const noexcept override { return message_.c_str(); }

  /**
   * @brief Get the maximum error code that occurred during evaluation.
   * @return The maximum error code
   */
  [[nodiscard]] errc error_code() const { return max_error_; }

  /**
   * @brief Get the per-row error codes (if available).
   * @return An optional device vector containing per-row error codes
   */
  [[nodiscard]] auto const& row_errors() const { return row_errors_; }

 private:
  errc max_error_;  //< The maximum error code that occurred during evaluation
  std::optional<rmm::device_uvector<errc>>
    row_errors_;         //< An optional device vector containing per-row error codes (if available)
  std::string message_;  //< An error message describing the evaluation error
};

}  // namespace CUDF_EXPORT cudf
