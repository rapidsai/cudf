/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS,  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#include <tests/iterator/iterator_tests.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/indexalator.cuh>

#include <thrust/host_vector.h>
#include <thrust/optional.h>
#include <thrust/pair.h>

using TestingTypes = cudf::test::IntegralTypesNotBool;

template <typename T>
struct IndexalatorTest : public IteratorTest<T> {
};

TYPED_TEST_SUITE(IndexalatorTest, TestingTypes);

TYPED_TEST(IndexalatorTest, input_iterator)
{
  using T = TypeParam;

  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});

  auto d_col = cudf::test::fixed_width_column_wrapper<T>(host_values.begin(), host_values.end());

  auto expected_values = thrust::host_vector<cudf::size_type>(host_values.size());
  std::transform(host_values.begin(), host_values.end(), expected_values.begin(), [](auto v) {
    return static_cast<cudf::size_type>(v);
  });

  auto it_dev = cudf::detail::indexalator_factory::make_input_iterator(d_col);
  this->iterator_test_thrust(expected_values, it_dev, host_values.size());
}

TYPED_TEST(IndexalatorTest, pair_iterator)
{
  using T = TypeParam;

  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -120, 115});
  auto validity    = std::vector<bool>({0, 1, 1, 1, 1, 1, 0, 1, 1});

  auto d_col = cudf::test::fixed_width_column_wrapper<T>(
    host_values.begin(), host_values.end(), validity.begin());

  auto expected_values =
    thrust::host_vector<thrust::pair<cudf::size_type, bool>>(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 validity.begin(),
                 expected_values.begin(),
                 [](T v, bool b) { return thrust::make_pair(static_cast<cudf::size_type>(v), b); });

  auto it_dev = cudf::detail::indexalator_factory::make_input_pair_iterator(d_col);
  this->iterator_test_thrust(expected_values, it_dev, host_values.size());
}

TYPED_TEST(IndexalatorTest, optional_iterator)
{
  using T = TypeParam;

  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -104, 103, 64, -13, -20, 45});
  auto validity    = std::vector<bool>({0, 1, 1, 1, 1, 1, 0, 1, 1});

  auto d_col = cudf::test::fixed_width_column_wrapper<T>(
    host_values.begin(), host_values.end(), validity.begin());

  auto expected_values = thrust::host_vector<thrust::optional<cudf::size_type>>(host_values.size());

  std::transform(host_values.begin(),
                 host_values.end(),
                 validity.begin(),
                 expected_values.begin(),
                 [](T v, bool b) {
                   return (b) ? thrust::make_optional(static_cast<cudf::size_type>(v))
                              : thrust::nullopt;
                 });

  auto it_dev = cudf::detail::indexalator_factory::make_input_optional_iterator(d_col);
  this->iterator_test_thrust(expected_values, it_dev, host_values.size());
}
