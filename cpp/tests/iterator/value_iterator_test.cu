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

auto strings_to_string_views(std::vector<std::string>& input_strings)
{
  auto all_valid = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
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

using TestingTypes = cudf::test::AllTypes;

TYPED_TEST_CASE(IteratorTest, TestingTypes);

// tests for non-null iterator (pointer of device array)
TYPED_TEST(IteratorTest, non_null_iterator)
{
  using T         = TypeParam;
  auto host_array = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});
  thrust::device_vector<T> dev_array(host_array);

  // calculate the expected value by CPU.
  thrust::host_vector<T> replaced_array(host_array);

  // driven by iterator as a pointer of device array.
  auto it_dev      = dev_array.begin();
  T expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  this->iterator_test_thrust(replaced_array, it_dev, dev_array.size());
  this->iterator_test_cub(expected_value, it_dev, dev_array.size());

  // test column input
  cudf::test::fixed_width_column_wrapper<T> w_col(host_array.begin(), host_array.end());
  this->values_equal_test(replaced_array, *cudf::column_device_view::create(w_col));
}

// Tests for null input iterator (column with null bitmap)
// Actually, we can use cub for reduction with nulls without creating custom
// kernel or multiple steps. We may accelerate the reduction for a column using
// cub
TYPED_TEST(IteratorTest, null_iterator)
{
  using T = TypeParam;
  T init  = T{0};
  // data and valid arrays
  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 1});

  // create a column with bool vector
  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  thrust::host_vector<T> replaced_array(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // TODO uncomment after time_point ostream operator<<
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev = cudf::detail::make_null_replacement_iterator(*d_col, T{0});
  this->iterator_test_cub(expected_value, it_dev, d_col->size());
  this->values_equal_test(replaced_array, *d_col);
}

struct TransformedIteratorTest : public IteratorTest<int8_t> {
};

// Tests up cast reduction with null iterator.
// The up cast iterator will be created by transform_iterator and
// cudf::detail::make_null_replacement_iterator(col, T{0})
TEST_F(TransformedIteratorTest, null_iterator_upcast)
{
  const int column_size{1000};
  using T        = int8_t;
  using T_upcast = int64_t;
  T init{0};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  cudf::test::UniformRandomGenerator<T> rng(-128, 127);
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
  std::generate(host_bools.begin(), host_bools.end(), [&rbg]() { return rbg.generate(); });

  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  thrust::host_vector<T> replaced_array(d_col->size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T_upcast expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev        = cudf::detail::make_null_replacement_iterator(*d_col, T{0});
  auto it_dev_upcast = thrust::make_transform_iterator(it_dev, thrust::identity<T_upcast>());
  this->iterator_test_thrust(replaced_array, it_dev_upcast, d_col->size());
  this->iterator_test_cub(expected_value, it_dev, d_col->size());
}

// Tests for square input iterator using helper strcut
// `cudf::transformer_squared<T, T_upcast>` The up cast iterator will be created
// by make_transform_iterator(
//        cudf::detail::make_null_replacement_iterator(col, T{0}),
//        cudf::detail::transformer_squared<T_upcast>)
TEST_F(TransformedIteratorTest, null_iterator_square)
{
  const int column_size{1000};
  using T        = int8_t;
  using T_upcast = int64_t;
  T init{0};
  cudf::transformer_squared<T_upcast> transformer{};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  cudf::test::UniformRandomGenerator<T> rng(-128, 127);
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
  std::generate(host_bools.begin(), host_bools.end(), [&rbg]() { return rbg.generate(); });

  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  thrust::host_vector<T_upcast> replaced_array(d_col->size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x * x : init; });
  T_upcast expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev         = cudf::detail::make_null_replacement_iterator(*d_col, T{0});
  auto it_dev_upcast  = thrust::make_transform_iterator(it_dev, thrust::identity<T_upcast>());
  auto it_dev_squared = thrust::make_transform_iterator(it_dev_upcast, transformer);
  this->iterator_test_thrust(replaced_array, it_dev_squared, d_col->size());
  this->iterator_test_cub(expected_value, it_dev_squared, d_col->size());
}

// TODO only few types
TEST_F(TransformedIteratorTest, large_size_reduction)
{
  using T = int64_t;

  const int column_size{1000000};
  const T init{0};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  cudf::test::UniformRandomGenerator<T> rng(-128, 128);
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
  std::generate(host_bools.begin(), host_bools.end(), [&rbg]() { return rbg.generate(); });

  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate by cudf::reduce
  thrust::host_vector<T> replaced_array(d_col->size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev = cudf::detail::make_null_replacement_iterator(*d_col, init);
  this->iterator_test_thrust(replaced_array, it_dev, d_col->size());
  this->iterator_test_cub(expected_value, it_dev, d_col->size());
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

TYPED_TEST(IteratorTest, error_handling)
{
  using T         = TypeParam;
  auto host_array = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 1});

  cudf::test::fixed_width_column_wrapper<T> w_col_no_null(host_array.begin(), host_array.end());
  cudf::test::fixed_width_column_wrapper<T> w_col_null(
    host_array.begin(), host_array.end(), host_bools.begin());
  auto d_col_no_null = cudf::column_device_view::create(w_col_no_null);
  auto d_col_null    = cudf::column_device_view::create(w_col_null);

  // expects error: data type mismatch
  if (!(std::is_same<T, double>::value)) {
    CUDF_EXPECT_THROW_MESSAGE((d_col_null->begin<double>()), "the data type mismatch");
  }
  // expects error: data type mismatch
  if (!(std::is_same<T, float>::value)) {
    CUDF_EXPECT_THROW_MESSAGE((cudf::detail::make_null_replacement_iterator(*d_col_null, float{0})),
                              "the data type mismatch");
  }

  CUDF_EXPECT_THROW_MESSAGE((cudf::detail::make_null_replacement_iterator(*d_col_no_null, T{0})),
                            "Unexpected non-nullable column.");

  CUDF_EXPECT_THROW_MESSAGE((d_col_no_null->pair_begin<T, true>()),
                            "Unexpected non-nullable column.");
  CUDF_EXPECT_NO_THROW((d_col_null->pair_begin<T, false>()));
  CUDF_EXPECT_NO_THROW((d_col_null->pair_begin<T, true>()));

  // scalar iterator
  using ScalarType = cudf::scalar_type_t<T>;
  std::unique_ptr<cudf::scalar> s(new ScalarType{T{1}, false});
  CUDF_EXPECT_THROW_MESSAGE((cudf::detail::make_scalar_iterator<T>(*s)),
                            "the scalar value must be valid");
  CUDF_EXPECT_NO_THROW((cudf::detail::make_pair_iterator<T>(*s)));
  // expects error: data type mismatch
  if (!(std::is_same<T, double>::value)) {
    CUDF_EXPECT_THROW_MESSAGE((cudf::detail::make_scalar_iterator<double>(*s)),
                              "the data type mismatch");
    CUDF_EXPECT_THROW_MESSAGE((cudf::detail::make_pair_iterator<double>(*s)),
                              "the data type mismatch");
  }
}

CUDF_TEST_PROGRAM_MAIN()
