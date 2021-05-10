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
#include <tests/iterator/optional_iterator_test.cuh>

using TestingTypes = cudf::test::NumericTypes;

template <typename T>
struct NumericOptionalIteratorTest : public IteratorTest<T> {
};

TYPED_TEST_CASE(NumericOptionalIteratorTest, TestingTypes);
TYPED_TEST(NumericOptionalIteratorTest, nonull_optional_iterator)
{
  nonull_optional_iterator(*this);
}
TYPED_TEST(NumericOptionalIteratorTest, null_optional_iterator) { null_optional_iterator(*this); }

// to print meanvar for debug.
template <typename T>
std::ostream& operator<<(std::ostream& os, cudf::meanvar<T> const& rhs)
{
  return os << "[" << rhs.value << ", " << rhs.value_squared << ", " << rhs.count << "] ";
};

// Transformers and Operators for optional_iterator test
template <typename ElementType>
struct transformer_optional_meanvar {
  using ResultType = thrust::optional<cudf::meanvar<ElementType>>;

  CUDA_HOST_DEVICE_CALLABLE
  ResultType operator()(thrust::optional<ElementType> const& optional)
  {
    if (optional.has_value()) {
      auto v = *optional;
      return cudf::meanvar<ElementType>{v, static_cast<ElementType>(v * v), 1};
    }
    return thrust::nullopt;
  }
};

struct sum_if_not_null {
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE thrust::optional<T> operator()(const thrust::optional<T>& lhs,
                                                           const thrust::optional<T>& rhs)
  {
    return lhs.value_or(T{0}) + rhs.value_or(T{0});
  }
};

// TODO: enable this test also at __CUDACC_DEBUG__
// This test causes fatal compilation error only at device debug mode.
// Workaround: exclude this test only at device debug mode.
#if !defined(__CUDACC_DEBUG__)
// This test computes `count`, `sum`, `sum_of_squares` at a single reduction call.
// It would be useful for `var`, `std` operation
TYPED_TEST(NumericOptionalIteratorTest, mean_var_output)
{
  using T        = TypeParam;
  using T_output = cudf::meanvar<T>;
  transformer_optional_meanvar<T> transformer{};

  const int column_size{50};
  const T init{0};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  cudf::test::UniformRandomGenerator<T> rng;
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
  std::generate(host_bools.begin(), host_bools.end(), [&rbg]() { return rbg.generate(); });

  cudf::test::fixed_width_column_wrapper<TypeParam> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate expected values by CPU
  T_output expected_value;

  expected_value.count = d_col->size() - static_cast<cudf::column_view>(w_col).null_count();

  std::vector<T> replaced_array(d_col->size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? static_cast<T>(x) : init; });

  expected_value.count = d_col->size() - static_cast<cudf::column_view>(w_col).null_count();
  expected_value.value = std::accumulate(replaced_array.begin(), replaced_array.end(), T{0});
  expected_value.value_squared = std::accumulate(
    replaced_array.begin(), replaced_array.end(), T{0}, [](T acc, T i) { return acc + i * i; });

  // std::cout << "expected <mixed_output> = " << expected_value << std::endl;

  // GPU test
  auto it_dev         = d_col->optional_begin<T>(cudf::contains_nulls::YES{});
  auto it_dev_squared = thrust::make_transform_iterator(it_dev, transformer);
  auto result         = thrust::reduce(it_dev_squared,
                               it_dev_squared + d_col->size(),
                               thrust::optional<T_output>{T_output{}},
                               sum_if_not_null{});
  if (not std::is_floating_point<T>()) {
    EXPECT_EQ(expected_value, *result) << "optional iterator reduction sum";
  } else {
    EXPECT_NEAR(expected_value.value, result->value, 1e-3) << "optional iterator reduction sum";
    EXPECT_NEAR(expected_value.value_squared, result->value_squared, 1e-3)
      << "optional iterator reduction sum squared";
    EXPECT_EQ(expected_value.count, result->count) << "optional iterator reduction count";
  }
}
#endif
