/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
 * @brief The interface for host-based UDF implementation.
 *
 * An implementation of host-based UDF needs to be derived from this base class, defining
 * its own version of the required functions. In particular:
 *  - The derived class is required to implement `get_empty_output`, `operator()`, `is_equal`,
 *    and `clone` functions.
 *  - If necessary, the derived class can also override `do_hash` to compute hashing for its
 *    instance, and `get_required_data` to selectively access to the input data as well as
 *    intermediate data provided by libcudf.
 *
 * Example of such implementation:
 * @code{.cpp}
 * struct my_udf_aggregation : cudf::host_udf_base {
 *   my_udf_aggregation() = default;
 *
 *   // This UDF aggregation needs `GROUPED_VALUES` and `GROUP_OFFSETS`,
 *   // and the result from groupby `MAX` aggregation.
 *   [[nodiscard]] data_attribute_set_t get_required_data() const override
 *   {
 *       return {groupby_data_attribute::GROUPED_VALUES,
 *               groupby_data_attribute::GROUP_OFFSETS,
 *               cudf::make_max_aggregation<cudf::groupby_aggregation>()};
 *   }
 *
 *   [[nodiscard]] output_t get_empty_output(
 *     [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
 *     [[maybe_unused]] rmm::cuda_stream_view stream,
 *     [[maybe_unused]] rmm::device_async_resource_ref mr) const override
 *   {
 *     // This UDF aggregation always returns a column of type INT32.
 *     return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
 *   }
 *
 *   [[nodiscard]] output_t operator()(input_map_t const& input,
 *                                     rmm::cuda_stream_view stream,
 *                                     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Perform UDF computation using the input data and return the result.
 *   }
 *
 *   [[nodiscard]] bool is_equal(host_udf_base const& other) const override
 *   {
 *     // Check if the other object is also instance of this class.
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
struct host_udf_base {
  host_udf_base()          = default;
  virtual ~host_udf_base() = default;

  /**
   * @brief Define the possible data needed for groupby aggregations.
   *
   * Note that only sort-based groupby aggregations are supported.
   */
  enum class groupby_data_attribute : int32_t {
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
   * @brief Describe possible data that may be needed in the derived class for its operations.
   *
   * Such data can be either intermediate data such as sorted values or group labels etc, or the
   * results of other aggregations.
   *
   * Each derived host-based UDF class may need a different set of data. It is inefficient to
   * evaluate and pass down all these possible data at once from libcudf. A solution for that is,
   * the derived class can define a subset of data that it needs and libcudf will evaluate
   * and pass down only data requested from that set.
   */
  struct data_attribute {
    /**
     * @brief Hold all possible data types for the input of the aggregation in the derived class.
     */
    using value_type = std::variant<groupby_data_attribute, std::unique_ptr<aggregation>>;
    value_type value;  ///< The actual data attribute, wrapped by this struct
                       ///< as a wrapper is needed to define `hash` and `equal_to` functors.

    data_attribute()                 = default;  ///< Default constructor
    data_attribute(data_attribute&&) = default;  ///< Move constructor

    /**
     * @brief Construct a new data attribute from an aggregation attribute.
     * @param value_ An aggregation attribute
     */
    template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, groupby_data_attribute>)>
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
   * avoid unnecessary computation performed in libcudf. If this function is not overridden, an
   * empty set is returned. That means all the data attributes (except results from other
   * aggregations in groupby) will be needed.
   *
   * @return A set of `data_attribute`
   */
  [[nodiscard]] virtual data_attribute_set_t get_required_data() const { return {}; }

  /**
   * @brief Hold all possible types of the data that is passed to the derived class for executing
   * the aggregation.
   */
  using input_data_t = std::variant<column_view, size_type, device_span<size_type const>>;

  /**
   * @brief Input to the aggregation, mapping from each data attribute to its actual data.
   */
  using input_map_t = std::
    unordered_map<data_attribute, input_data_t, data_attribute::hash, data_attribute::equal_to>;

  /**
   * @brief Output type of the aggregation.
   *
   * Currently only a single type is supported as the output of the aggregation, but it will hold
   * more type in the future when reduction is supported.
   */
  using output_t = std::variant<std::unique_ptr<column>>;

  /**
   * @brief Get the output when the input values column is empty.
   *
   * This is called in libcudf when the input values column is empty. In such situations libcudf
   * tries to generate the output directly without unnecessarily evaluating the intermediate data.
   *
   * @param output_dtype The expected output data type
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation when input values is empty
   */
  [[nodiscard]] virtual output_t get_empty_output(std::optional<data_type> output_dtype,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr) const = 0;

  /**
   * @brief Perform the main computation for the host-based UDF.
   *
   * @param input The input data needed for performing all computation
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation
   */
  [[nodiscard]] virtual output_t operator()(input_map_t const& input,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr) const = 0;

  /**
   * @brief Computes hash value of the class's instance.
   * @return The hash value of the instance
   */
  [[nodiscard]] virtual std::size_t do_hash() const
  {
    return std::hash<int>{}(static_cast<int>(aggregation::Kind::HOST_UDF));
  }

  /**
   * @brief Compares two instances of the derived class for equality.
   * @param other The other derived class's instance to compare with
   * @return True if the two instances are equal
   */
  [[nodiscard]] virtual bool is_equal(host_udf_base const& other) const = 0;

  /**
   * @brief Clones the instance.
   *
   * A class derived from `host_udf_base` should not store too much data such that its instances
   * remain lightweight for efficient cloning.
   *
   * @return A new instance cloned from this
   */
  [[nodiscard]] virtual std::unique_ptr<host_udf_base> clone() const = 0;
};

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
