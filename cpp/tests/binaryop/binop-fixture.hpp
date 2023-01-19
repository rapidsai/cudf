/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/detail/iterator.cuh>

#include <string>
#include <type_traits>

struct BinaryOperationTest : public cudf::test::BaseFixture {
  BinaryOperationTest() {}

  static constexpr int r_min = 1;
  static constexpr int r_max = 10;

  template <typename T>
  static auto make_data_iter(cudf::test::UniformRandomGenerator<T>& rand_gen)
  {
    return cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return rand_gen.generate(); });
  }

  static auto make_validity_iter()
  {
    cudf::test::UniformRandomGenerator<uint8_t> rand_gen(r_min, r_max);
    uint8_t mod_base = rand_gen.generate();
    return cudf::detail::make_counting_transform_iterator(
      0, [mod_base](auto row) { return (row % mod_base) > 0; });
  }

  template <typename T>
  static auto make_random_wrapped_column(cudf::size_type size)
  {
    cudf::test::UniformRandomGenerator<T> rand_gen(r_min, r_max);
    auto data_iter     = make_data_iter(rand_gen);
    auto validity_iter = make_validity_iter();

    return cudf::test::fixed_width_column_wrapper<T>(data_iter, data_iter + size, validity_iter);
  }

  template <typename T, std::enable_if_t<!std::is_same_v<T, std::string>>* = nullptr>
  auto make_random_wrapped_scalar()
  {
    cudf::test::UniformRandomGenerator<T> rand_gen(r_min, r_max);
    return cudf::scalar_type_t<T>(rand_gen.generate());
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, std::string>>* = nullptr>
  auto make_random_wrapped_scalar()
  {
    cudf::test::UniformRandomGenerator<uint8_t> rand_gen(r_min, r_max);
    uint8_t size = rand_gen.generate();
    std::string str{"ஔⒶbc⁂∰ൠ \tنж水✉♪✿™"};
    return cudf::scalar_type_t<T>(str.substr(0, size));
  }
};
