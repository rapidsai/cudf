/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/copy.h>

#include <iterator>

namespace cudf {

template <typename T>
class split_device_span_iterator;

/**
 * @brief A device span consisting of two separate device_spans acting as if they were part of a
 * single span. The first head.size() entries are served from the first span, the remaining
 * tail.size() entries are served from the second span.
 *
 * @tparam T The type of elements in the span.
 */
template <typename T>
class split_device_span {
 public:
  using element_type    = T;
  using value_type      = std::remove_cv<T>;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = T*;
  using iterator        = split_device_span_iterator<T>;
  using const_pointer   = T const*;
  using reference       = T&;
  using const_reference = T const&;

  split_device_span() = default;

  explicit constexpr split_device_span(device_span<T> head, device_span<T> tail = {})
    : _head{head}, _tail{tail}
  {
  }

  [[nodiscard]] constexpr reference operator[](size_type i) const
  {
    return i < _head.size() ? _head[i] : _tail[i - _head.size()];
  }

  [[nodiscard]] constexpr size_type size() const { return _head.size() + _tail.size(); }

  [[nodiscard]] constexpr device_span<T> head() const { return _head; }

  [[nodiscard]] constexpr device_span<T> tail() const { return _tail; }

  [[nodiscard]] constexpr iterator begin() const;

  [[nodiscard]] constexpr iterator end() const;

 private:
  device_span<T> _head;
  device_span<T> _tail;
};

/**
 * @brief A random access iterator indexing into a split_device_span.
 *
 * @tparam T The type of elements in the underlying span.
 */
template <typename T>
class split_device_span_iterator {
  using it = split_device_span_iterator;

 public:
  using size_type         = std::size_t;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using pointer           = value_type*;
  using reference         = value_type&;
  using iterator_category = std::random_access_iterator_tag;

  split_device_span_iterator() = default;

  constexpr split_device_span_iterator(split_device_span<T> span, size_type offset)
    : _span{span}, _offset{offset}
  {
  }

  [[nodiscard]] constexpr reference operator*() const { return _span[_offset]; }

  [[nodiscard]] constexpr reference operator[](size_type i) const { return _span[_offset + i]; }

  [[nodiscard]] constexpr friend bool operator==(const it& lhs, const it& rhs)
  {
    return lhs._offset == rhs._offset;
  }

  [[nodiscard]] constexpr friend bool operator!=(const it& lhs, const it& rhs)
  {
    return !(lhs == rhs);
  }
  [[nodiscard]] constexpr friend bool operator<(const it& lhs, const it& rhs)
  {
    return lhs._offset < rhs._offset;
  }

  [[nodiscard]] constexpr friend bool operator>=(const it& lhs, const it& rhs)
  {
    return !(lhs < rhs);
  }

  [[nodiscard]] constexpr friend bool operator>(const it& lhs, const it& rhs) { return rhs < lhs; }

  [[nodiscard]] constexpr friend bool operator<=(const it& lhs, const it& rhs)
  {
    return !(lhs > rhs);
  }

  [[nodiscard]] constexpr friend difference_type operator-(const it& lhs, const it& rhs)
  {
    return lhs._offset - rhs._offset;
  }

  [[nodiscard]] constexpr friend it operator+(it lhs, difference_type i) { return lhs += i; }

  constexpr it& operator+=(difference_type i)
  {
    _offset += i;
    return *this;
  }

  constexpr it& operator-=(difference_type i) { return *this += -i; }

  constexpr it& operator++() { return *this += 1; }

  constexpr it& operator--() { return *this -= 1; }

  constexpr it operator++(int)
  {
    auto result = *this;
    ++*this;
    return result;
  }

  constexpr it operator--(int)
  {
    auto result = *this;
    --*this;
    return result;
  }

 private:
  split_device_span<T> _span;
  size_type _offset;
};

template <typename T>
[[nodiscard]] constexpr split_device_span_iterator<T> split_device_span<T>::begin() const
{
  return {*this, 0};
}

template <typename T>
[[nodiscard]] constexpr split_device_span_iterator<T> split_device_span<T>::end() const
{
  return {*this, size()};
}

/**
 * @brief A chunked storage class that provides preallocated memory for algorithms with known
 * worst-case output size. It provides functionality to retrieve the next chunk to write to, for
 * reporting how much memory was actually written and for gathering all previously written outputs
 * into a single contiguous vector.
 *
 * @tparam T The output element type.
 */
template <typename T>
class output_builder {
 public:
  using size_type = typename rmm::device_uvector<T>::size_type;

  /**
   * @brief Initializes an output builder with given worst-case output size and stream.
   *
   * @param max_write_size the maximum number of elements that will be written into a
   *                       split_device_span returned from `next_output`.
   * @param stream the stream used to allocate the first chunk of memory.
   * @param mr optional, the memory resource to use for allocation.
   */
  output_builder(size_type max_write_size,
                 size_type max_growth,
                 rmm::cuda_stream_view stream,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : _size{0}, _max_write_size{max_write_size}, _max_growth{max_growth}
  {
    CUDF_EXPECTS(max_write_size > 0, "Internal error");
    _chunks.emplace_back(0, stream, mr);
    _chunks.back().reserve(max_write_size * 2, stream);
  }

  output_builder(output_builder&&)      = delete;
  output_builder(const output_builder&) = delete;
  output_builder& operator=(output_builder&&) = delete;
  output_builder& operator=(const output_builder&) = delete;

  /**
   * @brief Returns the next free chunk of `max_write_size` elements from the underlying storage.
   * Must be followed by a call to `advance_output` after the memory has been written to.
   *
   * @param stream The stream to allocate a new chunk of memory with, if necessary.
   *               This should be the stream that will write to the `split_device_span`.
   * @return A `split_device_span` starting directly after the last output and providing at least
   *         `max_write_size` entries of storage.
   */
  [[nodiscard]] split_device_span<T> next_output(rmm::cuda_stream_view stream)
  {
    auto head_it   = _chunks.end() - (_chunks.size() > 1 and _chunks.back().is_empty() ? 2 : 1);
    auto head_span = get_free_span(*head_it);
    if (head_span.size() >= _max_write_size) { return split_device_span<T>{head_span}; }
    if (head_it == _chunks.end() - 1) {
      // insert a new device_uvector of double size
      auto const next_chunk_size =
        std::min(_max_growth * _max_write_size, 2 * _chunks.back().capacity());
      _chunks.emplace_back(0, stream, _chunks.back().memory_resource());
      _chunks.back().reserve(next_chunk_size, stream);
    }
    auto tail_span = get_free_span(_chunks.back());
    CUDF_EXPECTS(head_span.size() + tail_span.size() >= _max_write_size, "Internal error");
    return split_device_span<T>{head_span, tail_span};
  }

  /**
   * @brief Advances the output sizes after a `split_device_span` returned from `next_output` was
   *        written to.
   *
   * @param actual_size The number of elements that were written to the result of the previous
   *                    `next_output` call.
   * @param stream The stream on which to resize the vectors. Since this function will not
   *               reallocate, this only changes the stream of the internally stored vectors,
   *               impacting their subsequent copy and destruction behavior.
   */
  void advance_output(size_type actual_size, rmm::cuda_stream_view stream)
  {
    CUDF_EXPECTS(actual_size <= _max_write_size, "Internal error");
    if (_chunks.size() < 2) {
      auto const new_size = _chunks.back().size() + actual_size;
      inplace_resize(_chunks.back(), new_size, stream);
    } else {
      auto& tail              = _chunks.back();
      auto& prev              = _chunks.rbegin()[1];
      auto const prev_advance = std::min(actual_size, prev.capacity() - prev.size());
      auto const tail_advance = actual_size - prev_advance;
      inplace_resize(prev, prev.size() + prev_advance, stream);
      inplace_resize(tail, tail.size() + tail_advance, stream);
    }
    _size += actual_size;
  }

  /**
   * @brief Returns the first element that was written to the output.
   *        Requires a previous call to `next_output` and `advance_output` and `size() > 0`.
   * @param stream The stream used to access the element.
   * @return The first element that was written to the output.
   */
  [[nodiscard]] T front_element(rmm::cuda_stream_view stream) const
  {
    return _chunks.front().front_element(stream);
  }

  /**
   * @brief Returns the last element that was written to the output.
   *        Requires a previous call to `next_output` and `advance_output` and `size() > 0`.
   * @param stream The stream used to access the element.
   * @return The last element that was written to the output.
   */
  [[nodiscard]] T back_element(rmm::cuda_stream_view stream) const
  {
    auto const& last_nonempty_chunk =
      _chunks.size() > 1 and _chunks.back().is_empty() ? _chunks.rbegin()[1] : _chunks.back();
    return last_nonempty_chunk.back_element(stream);
  }

  [[nodiscard]] size_type size() const { return _size; }

  /**
   * @brief Gathers all previously written outputs into a single contiguous vector.
   *
   * @param stream The stream used to allocate and gather the output vector. All previous write
   *               operations to the output buffer must have finished or happened on this stream.
   * @param mr The memory resource used to allocate the output vector.
   * @return The output vector.
   */
  rmm::device_uvector<T> gather(rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr) const
  {
    rmm::device_uvector<T> output{size(), stream, mr};
    auto output_it = output.begin();
    for (auto const& chunk : _chunks) {
      output_it = thrust::copy(
        rmm::exec_policy_nosync(stream), chunk.begin(), chunk.begin() + chunk.size(), output_it);
    }
    return output;
  }

 private:
  /**
   * @brief Resizes a vector without reallocating
   *
   * @param vector The vector
   * @param new_size The new size. Must be smaller than the vector's capacity
   * @param stream The stream on which to resize the vector. Since this function will not
   *               reallocate, this only changes the stream of `vector`, impacting its subsequent
   *               copy and destruction behavior.
   */
  static void inplace_resize(rmm::device_uvector<T>& vector,
                             size_type new_size,
                             rmm::cuda_stream_view stream)
  {
    CUDF_EXPECTS(new_size <= vector.capacity(), "Internal error");
    vector.resize(new_size, stream);
  }

  /**
   * @brief Returns the span consisting of all currently unused elements in the vector
   *        (`i >= size() and i < capacity()`).
   *
   * @param vector The vector.
   * @return The span of unused elements.
   */
  static device_span<T> get_free_span(rmm::device_uvector<T>& vector)
  {
    return device_span<T>{vector.data() + vector.size(), vector.capacity() - vector.size()};
  }

  size_type _size;
  size_type _max_write_size;
  size_type _max_growth;
  std::vector<rmm::device_uvector<T>> _chunks;
};

}  // namespace cudf
