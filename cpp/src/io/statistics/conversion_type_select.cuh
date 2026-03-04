/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file conversion_type_select.cuh
 * @brief Utility classes for timestamp and duration conversion for PARQUET and ORC
 */

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

namespace cudf {
namespace io {
namespace detail {

template <int, int, typename>
class DetectInnerIteration;

template <int N0, typename... T>
class DetectInnerIteration<N0, 0, std::tuple<T...>> {
 public:
  static constexpr bool is_duplicate =
    std::is_same_v<typename std::tuple_element<N0, std::tuple<T...>>::type,
                   typename std::tuple_element<0, std::tuple<T...>>::type>;
};

template <int N0, int N1, typename... T>
class DetectInnerIteration<N0, N1, std::tuple<T...>> {
 public:
  static constexpr bool is_duplicate =
    std::is_same_v<typename std::tuple_element<N0, std::tuple<T...>>::type,
                   typename std::tuple_element<N1, std::tuple<T...>>::type> ||
    DetectInnerIteration<N0, N1 - 1, std::tuple<T...>>::is_duplicate;
};

template <int, typename>
class DetectIteration;

template <typename... T>
class DetectIteration<0, std::tuple<T...>> {
 public:
  static constexpr bool is_duplicate = false;
};

template <int N, typename... T>
class DetectIteration<N, std::tuple<T...>> {
 public:
  static constexpr bool is_duplicate =
    DetectInnerIteration<N, N - 1, std::tuple<T...>>::is_duplicate ||
    DetectIteration<N - 1, std::tuple<T...>>::is_duplicate;
};

template <typename>
class Detect;

/**
 * @brief Utility class to detect multiple occurrences of a type in the first element of pairs in a
 * tuple For eg. with the following tuple :
 *
 * using conversion_types =
 * std::tuple<
 *  std::pair<int, A>,
 *  std::pair<char, B>,
 *  std::pair<int, C>,
 *  std::pair<int, D>,
 *  std::pair<unsigned, E>,
 *  std::pair<unsigned, F>>;
 *
 * Detect<conversion_types>::is_duplicate will evaluate to true at compile time.
 * Here std::pair<int, A>, std::pair<int, C> and std::pair<int, D> are treated as duplicates
 * and std::pair<unsigned, E> and std::pair<unsigned, F>> are treated as duplicates.
 *
 * @tparam T... Parameter pack of pairs of types
 */
template <typename... T>
class Detect<std::tuple<T...>> {
 public:
  static constexpr bool is_duplicate =
    DetectIteration<(sizeof...(T) - 1), std::tuple<T...>>::is_duplicate;
};

template <typename>
class ConversionTypeSelect;

template <typename I0>
class ConversionTypeSelect<std::tuple<I0>> {
 public:
  template <typename T>
  using type = std::conditional_t<std::is_same_v<T, typename std::tuple_element<0, I0>::type>,
                                  typename std::tuple_element<1, I0>::type,
                                  T>;
};

/**
 * @brief Utility to select between types based on an input type
 *
 * using Conversion = std::tuple<
 *  std::pair<cudf::timestamp_s, cudf::timestamp_ms>,
 *  std::pair<cudf::timestamp_ns, cudf::timestamp_us>,
 *  std::pair<cudf::duration_s, cudf::duration_ms>,
 *  std::pair<cudf::duration_ns, cudf::duration_us>>
 *
 * using type = ConversionTypeSelect<Conversion>::type<cudf::duration_ns>
 * Here type will resolve to cudf::duration_us
 * If the type passed does not match any entries the type is returned as it is
 * This utility takes advantage of Detect class to reject any tuple with duplicate first
 * entries at compile time
 *
 * @tparam T... Parameter pack of pairs of types
 */
template <typename I0, typename... In>
class ConversionTypeSelect<std::tuple<I0, In...>> {
 public:
  template <typename T>
  using type =
    std::conditional_t<std::is_same_v<T, typename std::tuple_element<0, I0>::type>,
                       typename std::tuple_element<1, I0>::type,
                       typename ConversionTypeSelect<std::tuple<In...>>::template type<T>>;

  static_assert(not Detect<std::tuple<I0, In...>>::is_duplicate,
                "Type tuple has duplicate first entries");
};

}  // namespace detail
}  // namespace io
}  // namespace cudf
