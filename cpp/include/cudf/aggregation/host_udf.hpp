/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <functional>
#include <optional>
#include <unordered_map>
#include <variant>

/**
 * @file host_udf.hpp
 * @brief Declare the base class for host-side user-defined function (`HOST_UDF`) and example of
 * subclass implementation.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup aggregation_factories
 * @{
 */

/**
 * @brief The fundamental interface for host-based UDF implementation.
 *
 * This class declares the functions `do_hash`, `is_equal`, and `clone` that must be defined in
 * the users' UDF implementation. These function are required for libcudf aggregation framework
 * to perform its operations.
 */
class host_udf_base {
  // Declare constructor private to prevent the users from deriving from this class.
 private:
  host_udf_base() = default;  ///< Default constructor

  // Only allow deriving from the structs below.
  friend struct host_udf_reduction_base;
  friend struct host_udf_segmented_reduction_base;
  friend struct host_udf_groupby_base;

 public:
  virtual ~host_udf_base() = default;  ///< Default destructor

  /**
   * @brief Computes hash value of the instance.
   *
   * Overriding this function is optional when the derived class has data members such that
   * each instance needs to be differentiated from each other.
   *
   * @return The hash value of the instance
   */
  [[nodiscard]] virtual std::size_t do_hash() const
  {
    return std::hash<int>{}(static_cast<int>(aggregation::Kind::HOST_UDF));
  }

  /**
   * @brief Compares two instances of the derived class for equality.
   * @param other The other instance to compare with
   * @return True if the two instances are equal
   */
  [[nodiscard]] virtual bool is_equal(host_udf_base const& other) const = 0;

  /**
   * @brief Clones the instance.
   *
   * The instances of the derived class should be lightweight for efficient cloning.
   *
   * @return A new instance cloned from this one
   */
  [[nodiscard]] virtual std::unique_ptr<host_udf_base> clone() const = 0;
};

/**
 * @brief The interface for host-based UDF implementation for reduction context.
 *
 * An implementation of host-based UDF for reduction needs to be derived from this class.
 * In addition to implementing the virtual functions declared in the base class `host_udf_base`,
 * such derived class must also define the `operator()` function to perform reduction
 * operations.
 *
 * Example:
 * @code{.cpp}
 * struct my_udf_aggregation : cudf::host_udf_reduction_base {
 *   my_udf_aggregation() = default;
 *
 *   [[nodiscard]] std::unique_ptr<scalar> operator()(
 *     column_view const& input,
 *     data_type output_dtype,
 *     std::optional<std::reference_wrapper<scalar const>> init,
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Perform reduction computation using the input data and return the reduction result.
 *     // This is where the actual reduction logic is implemented.
 *   }
 *
 *   [[nodiscard]] bool is_equal(host_udf_base const& other) const override
 *   {
 *     // Check if the other object is also instance of this class.
 *     // If there are internal state variables, they may need to be checked for equality as well.
 *     return dynamic_cast<my_udf_aggregation const*>(&other) != nullptr;
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
 *   {
 *     return std::make_unique<my_udf_aggregation>();
 *   }
 * };
 * @endcode
 */
struct host_udf_reduction_base : host_udf_base {
  /**
   * @brief Perform reduction operations.
   *
   * @param input The input column for reduction
   * @param output_dtype The data type for the final output scalar
   * @param init The initial value of the reduction
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation
   */
  [[nodiscard]] virtual std::unique_ptr<scalar> operator()(
    column_view const& input,
    data_type output_dtype,
    std::optional<std::reference_wrapper<scalar const>> init,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;
};

/**
 * @brief The interface for host-based UDF implementation for segmented reduction context.
 *
 * An implementation of host-based UDF for segmented reduction needs to be derived from this class.
 * In addition to implementing the virtual functions declared in the base class `host_udf_base`,
 * such derived class must also define the `operator()` function to perform segmented reduction.
 *
 * Example:
 * @code{.cpp}
 * struct my_udf_aggregation : cudf::host_udf_segmented_reduction_base {
 *   my_udf_aggregation() = default;
 *
 *   [[nodiscard]] std::unique_ptr<column> operator()(
 *     column_view const& input,
 *     device_span<size_type const> offsets,
 *     data_type output_dtype,
 *     null_policy null_handling,
 *     std::optional<std::reference_wrapper<scalar const>> init,
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Perform computation using the input data and return the result.
 *     // This is where the actual segmented reduction logic is implemented.
 *   }
 *
 *   [[nodiscard]] bool is_equal(host_udf_base const& other) const override
 *   {
 *     // Check if the other object is also instance of this class.
 *     // If there are internal state variables, they may need to be checked for equality as well.
 *     return dynamic_cast<my_udf_aggregation const*>(&other) != nullptr;
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
 *   {
 *     return std::make_unique<my_udf_aggregation>();
 *   }
 * };
 * @endcode
 */
struct host_udf_segmented_reduction_base : host_udf_base {
  /**
   * @brief Perform segmented reduction operations.
   *
   * @param input The input column for reduction
   * @param offsets A list of offsets defining the segments for reduction
   * @param output_dtype The data type for the final output column
   * @param init The initial value of the reduction
   * @param null_handling If `INCLUDE` then the reduction result is valid only if all elements in
   *        the segment are valid, and if `EXCLUDE` then the reduction result is valid if any
   *        element in the segment is valid
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation
   */
  [[nodiscard]] virtual std::unique_ptr<column> operator()(
    column_view const& input,
    device_span<size_type const> offsets,
    data_type output_dtype,
    null_policy null_handling,
    std::optional<std::reference_wrapper<scalar const>> init,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;
};

/**
 * @brief The interface for host-based UDF implementation for groupby aggregation context.
 *
 * An implementation of host-based UDF for groupby needs to be derived from this class.
 * In addition to implementing the virtual functions declared in the base class `host_udf_base`,
 * such derived class must also define the functions `get_empty_output` to return result when the
 * input is empty, and `operator()` to perform its groupby operations.
 *
 * @note Only sort-based groupby aggregations are supported at this time. Hash-based groupby
 * aggregations require more complex data structure and is not yet supported.
 *
 * Example:
 * @code{.cpp}
 * struct my_udf_aggregation : cudf::host_udf_groupby_base {
 *   my_udf_aggregation() = default;
 *
 *   [[nodiscard]] std::unique_ptr<column> get_empty_output(
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Return a column corresponding to the result when the input values column is empty.
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<column> operator()(
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Perform UDF computation using the input data and return the result.
 *     auto const values = get_data<groupby_data::GROUPED_VALUES>();
 *     auto const offsets = get_data<groupby_data::GROUP_OFFSETS>();
 *     auto const group_max = compute_aggregation(
 *         cudf::make_max_aggregation<cudf::groupby_aggregation>());
 *     ...
 *   }
 *
 *   [[nodiscard]] bool is_equal(host_udf_base const& other) const override
 *   {
 *     // Check if the other object is also instance of this class.
 *     // If there are internal state variables, they may need to be checked for equality as well.
 *     return dynamic_cast<my_udf_aggregation const*>(&other) != nullptr;
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
 *   {
 *     return std::make_unique<my_udf_aggregation>();
 *   }
 * };
 * @endcode
 */
struct host_udf_groupby_base : host_udf_base {
  /**
   * @brief Define the data that can be provided by libcudf for the aggregation to perform its
   * computation.
   */
  enum class groupby_data : int32_t {
    INPUT_VALUES,    ///< The input values column.
    GROUPED_VALUES,  ///< The input values grouped according to the input `keys` for which the
                     ///< values within each group maintain their original order.
    SORTED_GROUPED_VALUES,  ///< The input values grouped according to the input `keys` and
                            ///< sorted within each group.
    NUM_GROUPS,             ///< The number of groups (i.e., number of distinct keys).
    GROUP_OFFSETS,          ///< The offsets separating groups.
    GROUP_LABELS            ///< Group labels (which is also the same as group indices).
  };

  /**
   * @brief Hold all possible types of the data corresponding all values of `groupby_data`.
   */
  using groupby_data_t =
    std::variant<column_view, /* INPUT_VALUES, GROUPED_VALUES, SORTED_GROUPED_VALUES */
                 size_type,   /* NUM_GROUPS */
                 device_span<size_type const> /* GROUP_OFFSETS, GROUP_LABELS */
                 >;

  /**
   * @brief Store callbacks to access groupby data from libcudf.
   *
   * Only the data specified in `groupby_data` enum can be accessed through these callbacks.
   */
  std::unordered_map<groupby_data, std::function<groupby_data_t(void)>> data_assessor_callbacks;

  /**
   * @brief Define the conditional output type for the template function `get_data<>`
   * below, allowing to access different data types similar to `std::get<>`.
   */
  template <groupby_data attr>
  using data_t = std::conditional_t<
    attr == groupby_data::INPUT_VALUES || attr == groupby_data::GROUPED_VALUES ||
      attr == groupby_data::SORTED_GROUPED_VALUES,
    column_view,
    std::conditional_t<attr == groupby_data::NUM_GROUPS, size_type, device_span<size_type const>>>;

  /**
   * @brief Access groupby data from libcudf based on the given by the template parameter.
   *
   * @return The data corresponding to the given template parameter.
   */
  template <groupby_data attr>
  [[nodiscard]] data_t<attr> get_data() const;

  /**
   * @brief Callback to access the result from other groupby aggregations.
   */
  std::function<column_view(std::unique_ptr<aggregation>)> aggregation_assessor_callback;

  /**
   * @brief Perform computing a built-in groupby aggregation and access its result.
   *
   * This allows the derived class to call any other built-in groupby aggregations on the same input
   * values column and access the output for its operations.
   *
   * @param other_agg An arbitrary built-in groupby aggregation
   * @return A `column_view` object corresponding to the output result of the given aggregation
   */
  [[nodiscard]] column_view compute_aggregation(std::unique_ptr<aggregation> other_agg) const
  {
    CUDF_EXPECTS(aggregation_assessor_callback,
                 "Uninitialized callback for computing aggregation.");
    return aggregation_assessor_callback(std::move(other_agg));
  }

  /**
   * @brief Get the output when the input values column is empty.
   *
   * This is called in libcudf when the input values column is empty. In such situations libcudf
   * tries to generate the output directly without unnecessarily evaluating the intermediate data.
   *
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation when the input values column is empty
   */
  [[nodiscard]] virtual std::unique_ptr<column> get_empty_output(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const = 0;

  /**
   * @brief Perform the main groupby computation for the host-based UDF.
   *
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation
   */
  [[nodiscard]] virtual std::unique_ptr<column> operator()(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const = 0;
};

/**
 * @brief Map each `groupby_data` enum to its corresponding data type.
 */
#define MAP_GROUPBY_DATA_TYPE(attr, output_type)                                                  \
  template <>                                                                                     \
  [[nodiscard]] inline host_udf_groupby_base::data_t<host_udf_groupby_base::groupby_data::attr>   \
  host_udf_groupby_base::get_data<host_udf_groupby_base::groupby_data::attr>() const              \
  {                                                                                               \
    CUDF_EXPECTS(data_assessor_callbacks.count(groupby_data::attr) > 0,                           \
                 "Uninitialized data accessor callbacks.");                                       \
    auto const& data_accessor = data_assessor_callbacks.at(groupby_data::attr);                   \
    CUDF_EXPECTS(data_accessor, "Uninitialized accessor callback for data attribute " #attr "."); \
    return std::get<output_type>(data_accessor());                                                \
  }

MAP_GROUPBY_DATA_TYPE(INPUT_VALUES, column_view)
MAP_GROUPBY_DATA_TYPE(GROUPED_VALUES, column_view)
MAP_GROUPBY_DATA_TYPE(SORTED_GROUPED_VALUES, column_view)
MAP_GROUPBY_DATA_TYPE(NUM_GROUPS, cudf::size_type)
MAP_GROUPBY_DATA_TYPE(GROUP_OFFSETS, device_span<cudf::size_type const>)
MAP_GROUPBY_DATA_TYPE(GROUP_LABELS, device_span<cudf::size_type const>)

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
