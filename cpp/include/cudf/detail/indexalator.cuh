/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/normalizing_iterator.cuh>

#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>
#include <thrust/pair.h>

namespace cudf {
namespace detail {

/**
 * @brief The index normalizing input iterator.
 *
 * This is an iterator that can be used for index types (integers) without
 * requiring a type-specific instance. It can be used for any iterator
 * interface for reading an array of integer values of type
 * int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
 * Reading specific elements always return a `size_type` integer.
 *
 * Use the indexalator_factory to create an appropriate input iterator
 * from a column_view.
 *
 * Example input iterator usage.
 * @code
 *  auto begin = indexalator_factory::create_input_iterator(gather_map);
 *  auto end   = begin + gather_map.size();
 *  auto result = detail::gather( source, begin, end, IGNORE, stream, mr );
 * @endcode
 *
 * @code
 *  auto begin = indexalator_factory::create_input_iterator(indices);
 *  auto end   = begin + indices.size();
 *  auto result = thrust::find(thrust::device, begin, end, size_type{12} );
 * @endcode
 */
using input_indexalator = input_normalator<cudf::size_type>;

/**
 * @brief The index normalizing output iterator.
 *
 * This is an iterator that can be used for index types (integers) without
 * requiring a type-specific instance. It can be used for any iterator
 * interface for writing an array of integer values of type
 * int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
 * Setting specific elements always accept `size_type` integer values.
 *
 * Use the indexalator_factory to create an appropriate output iterator
 * from a mutable_column_view.
 *
 * Example output iterator usage.
 * @code
 *  auto result_itr = indexalator_factory::create_output_iterator(indices->mutable_view());
 *  thrust::lower_bound(rmm::exec_policy(stream),
 *                      input->begin<Element>(),
 *                      input->end<Element>(),
 *                      values->begin<Element>(),
 *                      values->end<Element>(),
 *                      result_itr,
 *                      thrust::less<Element>());
 * @endcode
 */
using output_indexalator = output_normalator<cudf::size_type>;

/**
 * @brief Use this class to create an indexalator instance.
 */
struct indexalator_factory {
  /**
   * @brief A type_dispatcher functor to create an input iterator from an indices column.
   */
  struct input_indexalator_fn {
    template <typename IndexType, std::enable_if_t<is_index_type<IndexType>()>* = nullptr>
    input_indexalator operator()(column_view const& indices)
    {
      return input_indexalator(indices.data<IndexType>(), indices.type());
    }
    template <typename IndexType,
              typename... Args,
              std::enable_if_t<not is_index_type<IndexType>()>* = nullptr>
    input_indexalator operator()(Args&&... args)
    {
      CUDF_FAIL("indices must be an index type");
    }
  };

  /**
   * @brief Use this class to create an indexalator to a scalar index.
   */
  struct input_indexalator_scalar_fn {
    template <typename IndexType, std::enable_if_t<is_index_type<IndexType>()>* = nullptr>
    input_indexalator operator()(scalar const& index)
    {
      // note: using static_cast<scalar_type_t<IndexType> const&>(index) creates a copy
      auto const scalar_impl = static_cast<scalar_type_t<IndexType> const*>(&index);
      return input_indexalator(scalar_impl->data(), index.type());
    }
    template <typename IndexType,
              typename... Args,
              std::enable_if_t<not is_index_type<IndexType>()>* = nullptr>
    input_indexalator operator()(Args&&... args)
    {
      CUDF_FAIL("scalar must be an index type");
    }
  };

  /**
   * @brief A type_dispatcher functor to create an output iterator from an indices column.
   */
  struct output_indexalator_fn {
    template <typename IndexType, std::enable_if_t<is_index_type<IndexType>()>* = nullptr>
    output_indexalator operator()(mutable_column_view const& indices)
    {
      return output_indexalator(indices.data<IndexType>(), indices.type());
    }
    template <typename IndexType,
              typename... Args,
              std::enable_if_t<not is_index_type<IndexType>()>* = nullptr>
    output_indexalator operator()(Args&&... args)
    {
      CUDF_FAIL("indices must be an index type");
    }
  };

  /**
   * @brief Create an input indexalator instance from an indices column.
   */
  static input_indexalator make_input_iterator(column_view const& indices)
  {
    return type_dispatcher(indices.type(), input_indexalator_fn{}, indices);
  }

  /**
   * @brief Create an input indexalator instance from an index scalar.
   */
  static input_indexalator make_input_iterator(cudf::scalar const& index)
  {
    return type_dispatcher(index.type(), input_indexalator_scalar_fn{}, index);
  }

  /**
   * @brief Create an output indexalator instance from an indices column.
   */
  static output_indexalator make_output_iterator(mutable_column_view const& indices)
  {
    return type_dispatcher(indices.type(), output_indexalator_fn{}, indices);
  }

  /**
   * @brief An index accessor that returns a validity flag along with the index value.
   *
   * This is suitable as a `pair_iterator` for calling functions like `copy_if_else`.
   */
  struct nullable_index_accessor {
    input_indexalator iter;
    bitmask_type const* null_mask{};
    size_type const offset{};
    bool const has_nulls{};

    /**
     * @brief Create an accessor from a column_view.
     */
    nullable_index_accessor(column_view const& col, bool has_nulls = false)
      : null_mask{col.null_mask()}, offset{col.offset()}, has_nulls{has_nulls}
    {
      if (has_nulls) { CUDF_EXPECTS(col.nullable(), "Unexpected non-nullable column."); }
      iter = make_input_iterator(col);
    }

    __device__ thrust::pair<size_type, bool> operator()(size_type i) const
    {
      return {iter[i], (has_nulls ? bit_is_set(null_mask, i + offset) : true)};
    }
  };

  /**
   * @brief An index accessor that returns a validity flag along with the index value.
   *
   * This is suitable as a `pair_iterator`.
   */
  struct scalar_nullable_index_accessor {
    input_indexalator iter;
    bool const is_null;

    /**
     * @brief Create an accessor from a scalar.
     */
    scalar_nullable_index_accessor(scalar const& input) : is_null{!input.is_valid()}
    {
      iter = indexalator_factory::make_input_iterator(input);
    }

    __device__ thrust::pair<size_type, bool> operator()(size_type) const
    {
      return {*iter, is_null};
    }
  };

  /**
   * @brief Create an index iterator with a nullable index accessor.
   */
  static auto make_input_pair_iterator(column_view const& col)
  {
    return make_counting_transform_iterator(0, nullable_index_accessor{col, col.has_nulls()});
  }

  /**
   * @brief Create an index iterator with a nullable index accessor for a scalar.
   */
  static auto make_input_pair_iterator(scalar const& input)
  {
    return thrust::make_transform_iterator(thrust::make_constant_iterator<size_type>(0),
                                           scalar_nullable_index_accessor{input});
  }

  /**
   * @brief An index accessor that returns an index value if corresponding validity flag is true.
   *
   * This is suitable as an `optional_iterator`.
   */
  struct optional_index_accessor {
    input_indexalator iter;
    bitmask_type const* null_mask{};
    size_type const offset{};
    bool const has_nulls{};

    /**
     * @brief Create an accessor from a column_view.
     */
    optional_index_accessor(column_view const& col, bool has_nulls = false)
      : null_mask{col.null_mask()}, offset{col.offset()}, has_nulls{has_nulls}
    {
      if (has_nulls) { CUDF_EXPECTS(col.nullable(), "Unexpected non-nullable column."); }
      iter = make_input_iterator(col);
    }

    __device__ thrust::optional<size_type> operator()(size_type i) const
    {
      return has_nulls && !bit_is_set(null_mask, i + offset) ? thrust::nullopt
                                                             : thrust::make_optional(iter[i]);
    }
  };

  /**
   * @brief An index accessor that returns an index value if the scalar's validity flag is true.
   *
   * This is suitable as an `optional_iterator`.
   */
  struct scalar_optional_index_accessor {
    input_indexalator iter;
    bool const is_null;

    /**
     * @brief Create an accessor from a scalar.
     */
    scalar_optional_index_accessor(scalar const& input) : is_null{!input.is_valid()}
    {
      iter = indexalator_factory::make_input_iterator(input);
    }

    __device__ thrust::optional<size_type> operator()(size_type) const
    {
      return is_null ? thrust::nullopt : thrust::make_optional(*iter);
    }
  };

  /**
   * @brief Create an index iterator with an optional index accessor.
   */
  static auto make_input_optional_iterator(column_view const& col)
  {
    return make_counting_transform_iterator(0, optional_index_accessor{col, col.has_nulls()});
  }

  /**
   * @brief Create an index iterator with an optional index accessor for a scalar.
   */
  static auto make_input_optional_iterator(scalar const& input)
  {
    return thrust::make_transform_iterator(thrust::make_constant_iterator<size_type>(0),
                                           scalar_optional_index_accessor{input});
  }
};

}  // namespace detail
}  // namespace cudf
