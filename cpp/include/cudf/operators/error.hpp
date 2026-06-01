/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief An enumeration of error codes that can occur during operations.
 */
enum class errc : cuda::std::int8_t { SUCCESS = 0, OVERFLOW = 1, DIVISION_BY_ZERO = 2 };

/**
 * @brief A type that represents the result of an operation, which can either be a value or an
 * error.
 *
 * @tparam T The type of the value.
 */
template <typename T>
struct result {
 private:
  errc error_;  //< The error code of the result, errc::SUCCESS if the operation was successful
  T value_;     //< The value of the result, only valid if error_ is errc::SUCCESS

 public:
  /**
   * @brief Constructs a result with a value.
   *
   * @param value The value of the result.
   */
  CUDF_HOST_DEVICE constexpr result(T value) : error_(errc::SUCCESS), value_(value) {}

  /**
   * @brief Constructs a result with an error.
   *
   * @param error The error code of the result.
   */
  CUDF_HOST_DEVICE constexpr result(errc error) : error_(error), value_() {}

  /**
   * @brief Checks if the result has an error.
   *
   * @return true if the result has an error, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool has_error() const
  {
    return error_ != errc::SUCCESS;
  }

  /**
   * @brief Checks if the result has a value.
   *
   * @return true if the result has a value, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool has_value() const { return !has_error(); }

  /**
   * @brief Returns true if the result has a value, false otherwise. This operator allows the result
   * to be used in boolean contexts, such as if statements.
   *
   * @return true if the result has a value, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr explicit operator bool() const { return has_value(); }

  /**
   * @brief Returns the value of the result. Behaviour is undefined if the result has an error (i.e.
   * value() is called on a result that has an error).
   *
   * @return The value of the result.
   */
  CUDF_HOST_DEVICE constexpr T const& value() const { return value_; }

  /**
   * @brief Returns the value of the result (if successful) or null (if it has an error value).
   *
   * @return The value of the result or null.
   */
  CUDF_HOST_DEVICE constexpr cuda::std::optional<T> value_or_null() const
  {
    if (error_ != errc::SUCCESS) { return {}; }
    return value_;
  }

  /**
   * @brief Returns the error code of the result. Behaviour is undefined if the result has a value
   * (i.e. error() is called on a result that does not have an error).
   *
   * @return The error code of the result.
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr errc error() const { return error_; }
};

// Helper variable template to detect if a type is a result type
template <typename T>
constexpr bool is_result = false;

// Specialization for result types
template <typename T>
constexpr bool is_result<result<T>> = true;

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
