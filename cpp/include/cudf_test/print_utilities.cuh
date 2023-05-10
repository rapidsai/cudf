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

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <type_traits>

namespace cudf::test::print {

constexpr int32_t hex_tag = 0;

template <int32_t TagT, typename T>
struct TaggedType {
  T v;
};

template <typename T>
using hex_t = TaggedType<hex_tag, T>;

/**
 * @brief Function object to transform a built-in type to a tagged type (e.g., in order to print
 * values from an iterator returning uint32_t as hex values)
 *
 * @tparam TaggedTypeT A TaggedType template specialisation
 */
template <typename TaggedTypeT>
struct ToTaggedType {
  template <typename T>
  CUDF_HOST_DEVICE TaggedTypeT operator()(T const& v) const
  {
    return TaggedTypeT{v};
  }
};

/**
 * @brief Returns an iterator that causes the values from \p it to be printed as hex values.
 *
 * @tparam InItT A random-access input iterator type
 * @param it A random-access input iterator t
 * @return
 */
template <typename InItT>
auto hex(InItT it)
{
  using value_t  = typename std::iterator_traits<InItT>::value_type;
  using tagged_t = hex_t<value_t>;
  return thrust::make_transform_iterator(it, ToTaggedType<tagged_t>{});
}

template <typename T, CUDF_ENABLE_IF(std::is_integral_v<T>&& std::is_signed_v<T>)>
CUDF_HOST_DEVICE void print_value(int32_t width, T arg)
{
  printf("%*d", width, arg);
}

template <typename T, CUDF_ENABLE_IF(std::is_integral_v<T>&& std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE void print_value(int32_t width, T arg)
{
  printf("%*d", width, arg);
}

CUDF_HOST_DEVICE void print_value(int32_t width, char arg) { printf("%*c", width, arg); }

template <typename T>
CUDF_HOST_DEVICE void print_value(int32_t width, hex_t<T> arg)
{
  printf("%*X", width, arg.v);
}

namespace detail {
template <typename T>
CUDF_HOST_DEVICE void print_values(int32_t width, char delimiter, T arg)
{
  print_value(width, arg);
}

template <typename T, typename... Ts>
CUDF_HOST_DEVICE void print_values(int32_t width, char delimiter, T arg, Ts... args)
{
  print_value(width, arg);
  if (delimiter) printf("%c", delimiter);
  print_values(width, delimiter, args...);
}

template <typename... Ts>
__global__ void print_array_kernel(std::size_t count, int32_t width, char delimiter, Ts... args)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (std::size_t i = 0; i < count; i++) {
      printf("%6lu: ", i);
      print_values(width, delimiter, args[i]...);
      printf("\n");
    }
  }
}
}  // namespace detail

/**
 * @brief Prints \p count elements from each of the given device-accessible iterators.
 *
 * @param count The number of items to print from each device-accessible iterator
 * @param stream The cuda stream to which the printing kernel shall be dispatched
 * @param args List of iterators to be printed
 */
template <typename... Ts>
void print_array(std::size_t count, rmm::cuda_stream_view stream, Ts... args)
{
  // The width to pad printed numbers to
  constexpr int32_t width = 6;

  // Delimiter used for separating values from subsequent iterators
  constexpr char delimiter = ',';

  // TODO we want this to compile to nothing dependnig on compiler flag, rather than runtime
  if (std::getenv("CUDA_DBG_DUMP") != nullptr) {
    detail::print_array_kernel<<<1, 1, 0, stream.value()>>>(count, width, delimiter, args...);
  }
}

}  // namespace cudf::test::print
