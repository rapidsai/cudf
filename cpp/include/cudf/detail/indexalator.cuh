/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/constant_iterator.h>

namespace cudf {
namespace detail {

/**
 * @brief The base class for the input or output index normalizing iterator.
 *
 * This implementation uses CTRP to define the `input_indexalator` and the
 * `output_indexalator` classes. This is so this class can manipulate the
 * uniquely typed subclass member variable `p_` directly without requiring
 * virtual functions since iterator instances will be copied to device memory.
 *
 * The base class mainly manages updating the `p_` member variable while the
 * subclasses handle accessing individual elements in device memory.
 *
 * @tparam T The derived class type for the iterator.
 */
template <class T>
struct base_indexalator {
  using difference_type   = ptrdiff_t;
  using value_type        = size_type;
  using pointer           = size_type*;
  using iterator_category = std::random_access_iterator_tag;

  base_indexalator()                        = default;
  base_indexalator(base_indexalator const&) = default;
  base_indexalator(base_indexalator&&)      = default;
  base_indexalator& operator=(base_indexalator const&) = default;
  base_indexalator& operator=(base_indexalator&&) = default;

  /**
   * @brief Prefix increment operator.
   */
  CUDA_HOST_DEVICE_CALLABLE T& operator++()
  {
    T& derived = static_cast<T&>(*this);
    derived.p_ += width_;
    return derived;
  }

  /**
   * @brief Postfix increment operator.
   */
  CUDA_HOST_DEVICE_CALLABLE T operator++(int)
  {
    T tmp{static_cast<T&>(*this)};
    operator++();
    return tmp;
  }

  /**
   * @brief Prefix decrement operator.
   */
  CUDA_HOST_DEVICE_CALLABLE T& operator--()
  {
    T& derived = static_cast<T&>(*this);
    derived.p_ -= width_;
    return derived;
  }

  /**
   * @brief Postfix decrement operator.
   */
  CUDA_HOST_DEVICE_CALLABLE T operator--(int)
  {
    T tmp{static_cast<T&>(*this)};
    operator--();
    return tmp;
  }

  /**
   * @brief Compound assignment by sum operator.
   */
  CUDA_HOST_DEVICE_CALLABLE T& operator+=(difference_type offset)
  {
    T& derived = static_cast<T&>(*this);
    derived.p_ += offset * width_;
    return derived;
  }

  /**
   * @brief Increment by offset operator.
   */
  CUDA_HOST_DEVICE_CALLABLE T operator+(difference_type offset) const
  {
    auto tmp = T{static_cast<T const&>(*this)};
    tmp.p_ += (offset * width_);
    return tmp;
  }

  /**
   * @brief Addition assignment operator.
   */
  CUDA_HOST_DEVICE_CALLABLE friend T operator+(difference_type offset, T const& rhs)
  {
    T tmp{rhs};
    tmp.p_ += (offset * rhs.width_);
    return tmp;
  }

  /**
   * @brief Compound assignment by difference operator.
   */
  CUDA_HOST_DEVICE_CALLABLE T& operator-=(difference_type offset)
  {
    T& derived = static_cast<T&>(*this);
    derived.p_ -= offset * width_;
    return derived;
  }

  /**
   * @brief Decrement by offset operator.
   */
  CUDA_HOST_DEVICE_CALLABLE T operator-(difference_type offset) const
  {
    auto tmp = T{static_cast<T const&>(*this)};
    tmp.p_ -= (offset * width_);
    return tmp;
  }

  /**
   * @brief Subtraction assignment operator.
   */
  CUDA_HOST_DEVICE_CALLABLE friend T operator-(difference_type offset, T const& rhs)
  {
    T tmp{rhs};
    tmp.p_ -= (offset * rhs.width_);
    return tmp;
  }

  /**
   * @brief Compute offset from iterator difference operator.
   */
  CUDA_HOST_DEVICE_CALLABLE difference_type operator-(T const& rhs) const
  {
    auto derived = static_cast<T const&>(*this);
    return (derived.p_ - rhs.p_) / width_;
  }

  /**
   * @brief Equals to operator.
   */
  CUDA_HOST_DEVICE_CALLABLE bool operator==(T const& rhs) const
  {
    return rhs.p_ == static_cast<T const&>(*this).p_;
  }
  /**
   * @brief Not equals to operator.
   */
  CUDA_HOST_DEVICE_CALLABLE bool operator!=(T const& rhs) const
  {
    return rhs.p_ != static_cast<T const&>(*this).p_;
  }
  /**
   * @brief Less than operator.
   */
  CUDA_HOST_DEVICE_CALLABLE bool operator<(T const& rhs) const
  {
    return static_cast<T const&>(*this).p_ < rhs.p_;
  }
  /**
   * @brief Greater than operator.
   */
  CUDA_HOST_DEVICE_CALLABLE bool operator>(T const& rhs) const
  {
    return static_cast<T const&>(*this).p_ > rhs.p_;
  }
  /**
   * @brief Less than or equals to operator.
   */
  CUDA_HOST_DEVICE_CALLABLE bool operator<=(T const& rhs) const
  {
    return static_cast<T const&>(*this).p_ <= rhs.p_;
  }
  /**
   * @brief Greater than or equals to operator.
   */
  CUDA_HOST_DEVICE_CALLABLE bool operator>=(T const& rhs) const
  {
    return static_cast<T const&>(*this).p_ >= rhs.p_;
  }

 protected:
  /**
   * @brief Constructor assigns width and type member variables for base class.
   */
  base_indexalator(int32_t width, data_type dtype) : width_(width), dtype_(dtype) {}

  int width_;        /// integer type width = 1,2,4, or 8
  data_type dtype_;  /// for type-dispatcher calls
};

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
 *  auto result = detail::gather( source, begin, end, IGNORE, mr, stream );
 * @endcode
 *
 * @code
 *  auto begin = indexalator_factory::create_input_iterator(indices);
 *  auto end   = begin + indices.size();
 *  auto result = thrust::find(thrust::device, begin, end, size_type{12} );
 * @endcode
 */
struct input_indexalator : base_indexalator<input_indexalator> {
  friend struct indexalator_factory;
  friend struct base_indexalator<input_indexalator>;  // for CTRP

  using reference = size_type const;  // this keeps STL and thrust happy

  input_indexalator()                         = default;
  input_indexalator(input_indexalator const&) = default;
  input_indexalator(input_indexalator&&)      = default;
  input_indexalator& operator=(input_indexalator const&) = default;
  input_indexalator& operator=(input_indexalator&&) = default;

  /**
   * @brief Indirection operator returns the value at the current iterator position.
   */
  CUDA_DEVICE_CALLABLE size_type operator*() const { return operator[](0); }

  /**
   * @brief Dispatch functor for resolving a size_type value from any index type.
   */
  struct index_as_size_type {
    template <typename T, std::enable_if_t<is_index_type<T>()>* = nullptr>
    __device__ size_type operator()(void const* tp)
    {
      return static_cast<size_type>(*static_cast<T const*>(tp));
    }
    template <typename T, std::enable_if_t<not is_index_type<T>()>* = nullptr>
    __device__ size_type operator()(void const* tp)
    {
      release_assert(false and "only index types are supported");
      return 0;
    }
  };
  /**
   * @brief Array subscript operator returns a value at the input
   * `idx` position as a `size_type` value.
   */
  CUDA_DEVICE_CALLABLE size_type operator[](size_type idx) const
  {
    void const* tp = p_ + (idx * width_);
    return type_dispatcher(dtype_, index_as_size_type{}, tp);
  }

 protected:
  /**
   * @brief Create an input index normalizing iterator.
   *
   * Use the indexalator_factory to create an iterator instance.
   *
   * @param data      Pointer to an integer array in device memory.
   * @param width     The width of the integer type (1, 2, 4, or 8)
   * @param data_type Index integer type of width `width`
   */
  input_indexalator(void const* data, int width, data_type dtype)
    : base_indexalator<input_indexalator>(width, dtype), p_{static_cast<char const*>(data)}
  {
  }

  char const* p_;  /// pointer to the integer data in device memory
};

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
 *  thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
 *                      input->begin<Element>(),
 *                      input->end<Element>(),
 *                      values->begin<Element>(),
 *                      values->end<Element>(),
 *                      result_itr,
 *                      thrust::less<Element>());
 * @endcode
 */
struct output_indexalator : base_indexalator<output_indexalator> {
  friend struct indexalator_factory;
  friend struct base_indexalator<output_indexalator>;  // for CTRP

  using reference = output_indexalator const&;  // required for output iterators

  output_indexalator()                          = default;
  output_indexalator(output_indexalator const&) = default;
  output_indexalator(output_indexalator&&)      = default;
  output_indexalator& operator=(output_indexalator const&) = default;
  output_indexalator& operator=(output_indexalator&&) = default;

  /**
   * @brief Indirection operator returns this iterator instance in order
   * to capture the `operator=(size_type)` calls.
   */
  CUDA_DEVICE_CALLABLE output_indexalator const& operator*() const { return *this; }

  /**
   * @brief Array subscript operator returns an iterator instance at the specified `idx` position.
   *
   * This allows capturing the subsequent `operator=(size_type)` call in this class.
   */
  CUDA_DEVICE_CALLABLE output_indexalator const operator[](size_type idx) const
  {
    output_indexalator tmp{*this};
    tmp.p_ += (idx * width_);
    return tmp;
  }

  /**
   * @brief Dispatch functor for setting the index value from a size_type value.
   */
  struct size_type_to_index {
    template <typename T, std::enable_if_t<is_index_type<T>()>* = nullptr>
    __device__ void operator()(void* tp, size_type const value)
    {
      (*static_cast<T*>(tp)) = static_cast<T>(value);
    }
    template <typename T, std::enable_if_t<not is_index_type<T>()>* = nullptr>
    __device__ void operator()(void* tp, size_type const value)
    {
      release_assert(false and "only index types are supported");
    }
  };

  /**
   * @brief Assign a size_type value to the current iterator position.
   */
  CUDA_DEVICE_CALLABLE output_indexalator const& operator=(size_type const value) const
  {
    void* tp = p_;
    type_dispatcher(dtype_, size_type_to_index{}, tp, value);
    return *this;
  }

 protected:
  /**
   * @brief Create an output index normalizing iterator.
   *
   * Use the indexalator_factory to create an iterator instance.
   *
   * @param data      Pointer to an integer array in device memory.
   * @param width     The width of the integer type (1, 2, 4, or 8)
   * @param data_type Index integer type of width `width`
   */
  output_indexalator(void* data, int width, data_type dtype)
    : base_indexalator<output_indexalator>(width, dtype), p_{static_cast<char*>(data)}
  {
  }

  char* p_;  /// pointer to the integer data in device memory
};

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
      return input_indexalator(indices.data<IndexType>(), sizeof(IndexType), indices.type());
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
      return input_indexalator(scalar_impl->data(), sizeof(IndexType), index.type());
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
      return output_indexalator(indices.data<IndexType>(), sizeof(IndexType), indices.type());
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
};

}  // namespace detail
}  // namespace cudf
