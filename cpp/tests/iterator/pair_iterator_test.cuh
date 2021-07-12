/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

template <typename T>
void nonull_pair_iterator(IteratorTest<T>& testFixture)
{
  // data and valid arrays
  auto host_values_std =
    cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});
  thrust::host_vector<T> host_values(host_values_std);

  // create a column
  cudf::test::fixed_width_column_wrapper<T> w_col(host_values.begin(), host_values.end());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  thrust::host_vector<thrust::pair<T, bool>> replaced_array(host_values.size());
  std::transform(host_values.begin(), host_values.end(), replaced_array.begin(), [](auto s) {
    return thrust::make_pair(s, true);
  });

  // GPU test
  auto it_dev = d_col->pair_begin<T, false>();
  testFixture.iterator_test_thrust(replaced_array, it_dev, host_values.size());
}

template <typename T>
void null_pair_iterator(IteratorTest<T>& testFixture)
{
  // data and valid arrays
  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});
  thrust::host_vector<bool> host_bools(std::vector<bool>({1, 1, 0, 1, 1, 1, 0, 1, 1}));

  // create a column with bool vector
  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  thrust::host_vector<thrust::pair<T, bool>> value_and_validity(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 value_and_validity.begin(),
                 [](auto s, auto b) {
                   return thrust::pair<T, bool>{s, b};
                 });
  thrust::host_vector<thrust::pair<T, bool>> value_all_valid(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 value_all_valid.begin(),
                 [](auto s, auto b) {
                   return thrust::pair<T, bool>{s, true};
                 });

  // GPU test
  auto it_dev = d_col->pair_begin<T, true>();
  testFixture.iterator_test_thrust(value_and_validity, it_dev, host_values.size());

  auto it_hasnonull_dev = d_col->pair_begin<T, false>();
  testFixture.iterator_test_thrust(value_all_valid, it_hasnonull_dev, host_values.size());

  auto itb_dev = cudf::detail::make_validity_iterator(*d_col);
  testFixture.iterator_test_thrust(host_bools, itb_dev, host_values.size());
}
