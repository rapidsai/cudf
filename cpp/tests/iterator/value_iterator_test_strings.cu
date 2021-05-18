/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

auto strings_to_string_views(std::vector<std::string>& input_strings)
{
  auto all_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
  std::vector<char> chars;
  std::vector<int32_t> offsets;
  std::tie(chars, offsets) = cudf::test::detail::make_chars_and_offsets(
    input_strings.begin(), input_strings.end(), all_valid);
  thrust::device_vector<char> dev_chars(chars);
  char* c_start = thrust::raw_pointer_cast(dev_chars.data());

  // calculate the expected value by CPU. (but contains device pointers)
  std::vector<cudf::string_view> replaced_array(input_strings.size());
  std::transform(thrust::counting_iterator<size_t>(0),
                 thrust::counting_iterator<size_t>(replaced_array.size()),
                 replaced_array.begin(),
                 [c_start, offsets](auto i) {
                   return cudf::string_view(c_start + offsets[i], offsets[i + 1] - offsets[i]);
                 });
  return std::make_tuple(std::move(dev_chars), replaced_array);
}

struct StringIteratorTest : public IteratorTest<cudf::string_view> {
};

TEST_F(StringIteratorTest, string_view_null_iterator)
{
  using T = cudf::string_view;
  // T init = T{"", 0};
  std::string zero("zero");
  // the char data has to be in GPU
  thrust::device_vector<char> initmsg(zero.begin(), zero.end());
  T init = T{initmsg.data().get(), int(initmsg.size())};

  // data and valid arrays
  std::vector<std::string> host_values(
    {"one", "two", "three", "four", "five", "six", "eight", "nine"});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 1});

  // replace nulls in CPU
  std::vector<std::string> replaced_strings(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_strings.begin(),
                 [zero](auto s, auto b) { return b ? s : zero; });

  thrust::device_vector<char> dev_chars;
  thrust::host_vector<T> replaced_array(host_values.size());
  std::tie(dev_chars, replaced_array) = strings_to_string_views(replaced_strings);

  // create a column with bool vector
  cudf::test::strings_column_wrapper w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // GPU test
  auto it_dev = cudf::detail::make_null_replacement_iterator(*d_col, init);
  this->iterator_test_thrust(replaced_array, it_dev, host_values.size());
  // this->values_equal_test(replaced_array, *d_col); //string_view{0} is invalid
}

TEST_F(StringIteratorTest, string_view_no_null_iterator)
{
  using T = cudf::string_view;
  // T init = T{"", 0};
  std::string zero("zero");
  // the char data has to be in GPU
  thrust::device_vector<char> initmsg(zero.begin(), zero.end());
  T init = T{initmsg.data().get(), int(initmsg.size())};

  // data array
  std::vector<std::string> host_values(
    {"one", "two", "three", "four", "five", "six", "eight", "nine"});

  thrust::device_vector<char> dev_chars;
  thrust::host_vector<T> all_array(host_values.size());
  std::tie(dev_chars, all_array) = strings_to_string_views(host_values);

  // create a column with bool vector
  cudf::test::strings_column_wrapper w_col(host_values.begin(), host_values.end());
  auto d_col = cudf::column_device_view::create(w_col);

  // GPU test
  auto it_dev = d_col->begin<T>();
  this->iterator_test_thrust(all_array, it_dev, host_values.size());
}

TEST_F(StringIteratorTest, string_scalar_iterator)
{
  using T = cudf::string_view;
  // T init = T{"", 0};
  std::string zero("zero");
  // the char data has to be in GPU
  thrust::device_vector<char> initmsg(zero.begin(), zero.end());
  T init = T{initmsg.data().get(), int(initmsg.size())};

  // data array
  std::vector<std::string> host_values(100, zero);

  thrust::device_vector<char> dev_chars;
  thrust::host_vector<T> all_array(host_values.size());
  std::tie(dev_chars, all_array) = strings_to_string_views(host_values);

  // calculate the expected value by CPU.
  thrust::host_vector<thrust::pair<T, bool>> value_and_validity(host_values.size());
  std::transform(all_array.begin(), all_array.end(), value_and_validity.begin(), [](auto v) {
    return thrust::pair<T, bool>{v, true};
  });

  // create a scalar
  using ScalarType = cudf::scalar_type_t<T>;
  std::unique_ptr<cudf::scalar> s(new ScalarType{zero, true});

  // GPU test
  auto it_dev = cudf::detail::make_scalar_iterator<T>(*s);
  this->iterator_test_thrust(all_array, it_dev, host_values.size());

  auto it_pair_dev = cudf::detail::make_pair_iterator<T>(*s);
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());
}
