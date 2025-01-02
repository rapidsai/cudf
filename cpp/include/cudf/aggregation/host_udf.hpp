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
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <optional>
#include <unordered_map>
#include <unordered_set>
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
 * input is empty, and `operator()` to perform its groupby operations. Optionally, the derived
 * class can override the `get_required_data` function to selectively specify what data is
 * needed for its operations. Only the required data will be evaluated in libcudf to avoid
 * unnecessary work.
 *
 * @note Only sort-based groupby aggregations are supported at this time. Hash-based groupby
 * aggregations require more complex data structure and is not yet supported.
 *
 * Example:
 * @code{.cpp}
 * struct my_udf_aggregation : cudf::host_udf_groupby_base {
 *   my_udf_aggregation() = default;
 *
 *   [[nodiscard]] data_attribute_set_t get_required_data() const override
 *   {
 *       // This UDF aggregation needs `GROUPED_VALUES` and `GROUP_OFFSETS`,
 *       // and the result from groupby `MAX` aggregation.
 *       return {data_attribute::GROUPED_VALUES,
 *               data_attribute::GROUP_OFFSETS,
 *               cudf::make_max_aggregation<cudf::groupby_aggregation>()};
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<column> get_empty_output(
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Return a column corresponding to the result when the input values column is empty.
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<column> operator()(
 *     input_map_t const& input,
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Perform UDF computation using the input data and return the result.
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
   * @brief Describe possible data that may be needed in the derived class for its operations.
   *
   * Such data can be either general groupby data such as sorted values or group labels etc, or the
   * results of other groupby aggregations.
   *
   * Each derived host-based UDF class may need a different set of data. It is inefficient to
   * evaluate and pass down all these possible data at once from libcudf. A solution for that is,
   * the derived class can define a subset of data that it needs and libcudf will evaluate
   * and pass down only the data requested from that set.
   */
  struct data_attribute {
    /**
     * @brief Define the data that can be provided by libcudf for any sort-based groupby
     * aggregation.
     */
    enum general_attribute : int32_t {
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
     * @brief Hold all possible data types for the input of the aggregation in the derived class.
     */
    using value_type = std::variant<general_attribute, std::unique_ptr<aggregation>>;
    value_type value;  ///< The actual data attribute, wrapped by this struct
                       ///< as a wrapper is needed to define `hash` and `equal_to` functors.

    data_attribute()                 = default;  ///< Default constructor
    data_attribute(data_attribute&&) = default;  ///< Move constructor

    /**
     * @brief Construct a new data attribute from an aggregation attribute.
     * @param value_ An aggregation attribute
     */
    template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, general_attribute>)>
    data_attribute(T value_) : value{value_}
    {
    }

    /**
     * @brief Construct a new data attribute from another aggregation request.
     * @param value_ An aggregation request
     */
    template <typename T,
              CUDF_ENABLE_IF(std::is_same_v<T, aggregation> ||
                             std::is_same_v<T, groupby_aggregation>)>
    data_attribute(std::unique_ptr<T> value_) : value{std::move(value_)}
    {
      CUDF_EXPECTS(std::get<std::unique_ptr<aggregation>>(value) != nullptr,
                   "Invalid aggregation request.");
      if constexpr (std::is_same_v<T, aggregation>) {
        CUDF_EXPECTS(
          dynamic_cast<groupby_aggregation*>(std::get<std::unique_ptr<T>>(value).get()) != nullptr,
          "Requesting results from other aggregations is only supported in groupby "
          "aggregations.");
      }
    }

    /**
     * @brief Copy constructor.
     * @param other The other data attribute to copy from
     */
    data_attribute(data_attribute const& other);

    /**
     * @brief Hash functor for `data_attribute`.
     */
    struct hash {
      /**
       * @brief Compute the hash value of a data attribute.
       * @param attr The data attribute to hash
       * @return The hash value of the data attribute
       */
      std::size_t operator()(data_attribute const& attr) const;
    };  // struct hash

    /**
     * @brief Equality comparison functor for `data_attribute`.
     */
    struct equal_to {
      /**
       * @brief Check if two data attributes are equal.
       * @param lhs The left-hand side data attribute
       * @param rhs The right-hand side data attribute
       * @return True if the two data attributes are equal
       */
      bool operator()(data_attribute const& lhs, data_attribute const& rhs) const;
    };  // struct equal_to
  };    // struct data_attribute

  /**
   * @brief Set of attributes for the input data that is needed for computing the aggregation.
   */
  using data_attribute_set_t =
    std::unordered_set<data_attribute, data_attribute::hash, data_attribute::equal_to>;

  /**
   * @brief Return a set of attributes for the data that is needed for computing the aggregation.
   *
   * The derived class should return the attributes corresponding to only the data that it needs to
   * avoid unnecessary computation performed in libcudf. Data attributes can include either general
   * data attributes, or output results from other groupby aggregations.
   *
   * If this function is not overridden, an empty set is returned. This is equivalent to returning
   * a set containing all the general data attributes.
   *
   * @return A set of `data_attribute`, empty set means all data corresponding to every general
   *         data attribute will be needed
   */
  [[nodiscard]] virtual data_attribute_set_t get_required_data() const { return {}; }

  /**
   * @brief Hold all possible types of the data that is passed to the derived class for executing
   * the aggregation.
   */
  using input_data_t =
    std::variant<column_view, /* INPUT_VALUES, GROUPED_VALUES, SORTED_GROUPED_VALUES */
                 size_type,   /* NUM_GROUPS */
                 device_span<size_type const> /* GROUP_OFFSETS, GROUP_LABELS */
                 >;

  /**
   * @brief Input to the aggregation, mapping from each data attribute to its actual data.
   *
   * Note that we must explicitly specify the prefix `cudf::` to differentiate `cudf::size_type`
   * from the internal type `std::unordered_map::size_type`.
   */
  struct input_map_t : std::unordered_map<data_attribute,
                                          input_data_t,
                                          data_attribute::hash,
                                          data_attribute::equal_to> {
    /**
     * @brief Define the conditional output type for the template function `get<>` below, allowing
     * to retrieve different data types similar to `std::get<>`.
     */
    template <data_attribute::general_attribute attr>
    using output_t = std::conditional_t<attr == data_attribute::INPUT_VALUES ||
                                          attr == data_attribute::GROUPED_VALUES ||
                                          attr == data_attribute::SORTED_GROUPED_VALUES,
                                        column_view,
                                        std::conditional_t<attr == data_attribute::NUM_GROUPS,
                                                           cudf::size_type,
                                                           device_span<cudf::size_type const>>>;

    /**
     * @brief Reconstruct different data types from values stored in the internal hash map based on
     * `general_attribute` value given by the template parameter.
     *
     * @note The given parameter value must be requested in the data attribute set returned by
     * `get_required_data` function. Otherwise, `std::out_of_range` exception will occur.
     *
     * @return The data corresponding to the given template parameter.
     */
    template <data_attribute::general_attribute attr>
    output_t<attr> const& get() const;

    /**
     * @brief Retrieve the output result corresponding to some groupby aggregation.
     *
     * @note The given aggregation must be requested in the data attribute set returned by
     * `get_required_data` function. Otherwise, `std::out_of_range` exception will occur.
     *
     * @param aggregation A groupby aggregation corresponding to the requested data
     * @return A `column_view` object storing the output result of the given aggregation
     */
    column_view const& get(std::unique_ptr<aggregation> aggregation) const
    {
      return std::get<column_view>(at(std::move(aggregation)));
    }
  };

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
   * @brief Perform the main computation for the host-based UDF.
   *
   * @param input The input data needed for performing all computation
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation
   */
  [[nodiscard]] virtual std::unique_ptr<column> operator()(
    input_map_t const& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;
};

/**
 * @brief Map each data attribute enum to its data type.
 */
#define MAP_ATTRIBUTE_GROUPBY(attr, output_type)                                               \
  template <>                                                                                  \
  [[nodiscard]] inline host_udf_groupby_base::input_map_t::output_t<                           \
    host_udf_groupby_base::data_attribute::attr> const&                                        \
  host_udf_groupby_base::input_map_t::get<host_udf_groupby_base::data_attribute::attr>() const \
  {                                                                                            \
    return std::get<output_type>(at(data_attribute::attr));                                    \
  }

MAP_ATTRIBUTE_GROUPBY(INPUT_VALUES, column_view)
MAP_ATTRIBUTE_GROUPBY(GROUPED_VALUES, column_view)
MAP_ATTRIBUTE_GROUPBY(SORTED_GROUPED_VALUES, column_view)
MAP_ATTRIBUTE_GROUPBY(NUM_GROUPS, cudf::size_type)
MAP_ATTRIBUTE_GROUPBY(GROUP_OFFSETS, device_span<cudf::size_type const>)
MAP_ATTRIBUTE_GROUPBY(GROUP_LABELS, device_span<cudf::size_type const>)

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
