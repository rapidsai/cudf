/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

template <typename T, typename InputIterator>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns(cudf::size_type start,
                                                               cudf::size_type size,
                                                               InputIterator valids)
{
  auto iter = cudf::detail::make_counting_transform_iterator(start, [](auto i) { return T(i); });

  return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size, valids);
}

template <typename T, typename InputIterator>
cudf::table create_fixed_table(cudf::size_type num_cols,
                               cudf::size_type start,
                               cudf::size_type col_size,
                               InputIterator valids)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (int idx = 0; idx < num_cols; idx++) {
    cudf::test::fixed_width_column_wrapper<T> wrap =
      create_fixed_columns<T>(start + (idx * num_cols), col_size, valids);
    cols.push_back(wrap.release());
  }
  return cudf::table(std::move(cols));
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(
  std::vector<cudf::size_type> const& indices, bool nullable)
{
  std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    auto iter =
      cudf::detail::make_counting_transform_iterator(indices[index], [](auto i) { return T(i); });
    if (not nullable) {
      result.push_back(cudf::test::fixed_width_column_wrapper<T>(
        iter, iter + (indices[index + 1] - indices[index])));
    } else {
      auto valids = cudf::detail::make_counting_transform_iterator(
        indices[index], [](auto i) { return i % 2 == 0; });
      result.push_back(cudf::test::fixed_width_column_wrapper<T>(
        iter, iter + (indices[index + 1] - indices[index]), valids));
    }
  }

  return result;
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(
  std::vector<cudf::size_type> const& indices, std::vector<bool> const& validity)
{
  std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    auto iter =
      cudf::detail::make_counting_transform_iterator(indices[index], [](auto i) { return T(i); });
    result.push_back(cudf::test::fixed_width_column_wrapper<T>(
      iter, iter + (indices[index + 1] - indices[index]), validity.begin() + indices[index]));
  }

  return result;
}

template <typename T, typename ElementIter>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(
  std::vector<cudf::size_type> const& indices, ElementIter begin, bool nullable)
{
  std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    auto iter = begin + indices[index];

    if (not nullable) {
      result.push_back(cudf::test::fixed_width_column_wrapper<T>(
        iter, iter + (indices[index + 1] - indices[index])));
    } else {
      auto valids = cudf::detail::make_counting_transform_iterator(
        indices[index], [](auto i) { return i % 2 == 0; });
      result.push_back(cudf::test::fixed_width_column_wrapper<T>(
        iter, iter + (indices[index + 1] - indices[index]), valids));
    }
  }

  return result;
}

template <typename T, typename ElementIter>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(
  std::vector<cudf::size_type> const& indices, ElementIter begin, std::vector<bool> const& validity)
{
  std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    auto iter = begin + indices[index];
    result.push_back(cudf::test::fixed_width_column_wrapper<T>(
      iter, iter + (indices[index + 1] - indices[index]), validity.begin() + indices[index]));
  }

  return result;
}

template <typename T>
std::vector<cudf::table> create_expected_tables(cudf::size_type num_cols,
                                                std::vector<cudf::size_type> const& indices,
                                                bool nullable)
{
  std::vector<cudf::table> result;

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    std::vector<std::unique_ptr<cudf::column>> cols = {};

    for (int idx = 0; idx < num_cols; idx++) {
      auto iter = cudf::detail::make_counting_transform_iterator(indices[index] + (idx * num_cols),
                                                                 [](auto i) { return T(i); });

      if (not nullable) {
        cudf::test::fixed_width_column_wrapper<T> wrap(
          iter, iter + (indices[index + 1] - indices[index]));
        cols.push_back(wrap.release());
      } else {
        auto valids = cudf::detail::make_counting_transform_iterator(
          indices[index], [](auto i) { return i % 2 == 0; });
        cudf::test::fixed_width_column_wrapper<T> wrap(
          iter, iter + (indices[index + 1] - indices[index]), valids);
        cols.push_back(wrap.release());
      }
    }

    result.emplace_back(std::move(cols));
  }

  return result;
}

inline std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns(
  std::vector<std::string> const& strings,
  std::vector<cudf::size_type> const& indices,
  bool nullable)
{
  std::vector<cudf::test::strings_column_wrapper> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    if (not nullable) {
      result.emplace_back(strings.begin() + indices[index], strings.begin() + indices[index + 1]);
    } else {
      auto valids = cudf::detail::make_counting_transform_iterator(
        indices[index], [](auto i) { return i % 2 == 0; });
      result.emplace_back(
        strings.begin() + indices[index], strings.begin() + indices[index + 1], valids);
    }
  }

  return result;
}

inline std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns(
  std::vector<std::string> const& strings,
  std::vector<cudf::size_type> const& indices,
  std::vector<bool> const& validity)
{
  std::vector<cudf::test::strings_column_wrapper> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    result.emplace_back(strings.begin() + indices[index],
                        strings.begin() + indices[index + 1],
                        validity.begin() + indices[index]);
  }

  return result;
}

inline std::vector<cudf::table> create_expected_string_tables(
  std::vector<std::vector<std::string>> const strings,
  std::vector<cudf::size_type> const& indices,
  bool nullable)
{
  std::vector<cudf::table> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    std::vector<std::unique_ptr<cudf::column>> cols = {};

    for (int idx = 0; idx < 2; idx++) {
      if (not nullable) {
        cudf::test::strings_column_wrapper wrap(strings[idx].begin() + indices[index],
                                                strings[idx].begin() + indices[index + 1]);
        cols.push_back(wrap.release());
      } else {
        auto valids = cudf::detail::make_counting_transform_iterator(
          indices[index], [](auto i) { return i % 2 == 0; });
        cudf::test::strings_column_wrapper wrap(
          strings[idx].begin() + indices[index], strings[idx].begin() + indices[index + 1], valids);
        cols.push_back(wrap.release());
      }
    }

    result.emplace_back(std::move(cols));
  }

  return result;
}
