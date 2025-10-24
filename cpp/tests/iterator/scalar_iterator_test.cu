/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/iterator_tests.cuh>

#include <cudf_test/random.hpp>

#include <thrust/host_vector.h>
#include <thrust/pair.h>

using TestingTypes = cudf::test::FixedWidthTypesWithoutFixedPoint;

TYPED_TEST_SUITE(IteratorTest, TestingTypes);

TYPED_TEST(IteratorTest, scalar_iterator)
{
  using T = TypeParam;
  T init  = cudf::test::make_type_param_scalar<T>(
    cudf::test::UniformRandomGenerator<int>(-128, 128).generate());
  // data and valid arrays
  thrust::host_vector<T> host_values(100, init);
  std::vector<bool> host_bools(100, true);

  // create a scalar
  using ScalarType = cudf::scalar_type_t<T>;
  std::unique_ptr<cudf::scalar> s(new ScalarType{init, true});

  // calculate the expected value by CPU.
  thrust::host_vector<thrust::pair<T, bool>> value_and_validity(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 value_and_validity.begin(),
                 [](auto v, auto b) { return thrust::pair<T, bool>{v, b}; });

  // GPU test
  auto it_dev = cudf::detail::make_scalar_iterator<T>(*s);
  this->iterator_test_thrust(host_values, it_dev, host_values.size());

  auto it_pair_dev = cudf::detail::make_pair_iterator<T>(*s);
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());
}

TYPED_TEST(IteratorTest, null_scalar_iterator)
{
  using T = TypeParam;
  T init  = cudf::test::make_type_param_scalar<T>(
    cudf::test::UniformRandomGenerator<int>(-128, 128).generate());
  // data and valid arrays
  std::vector<T> host_values(100, init);
  std::vector<bool> host_bools(100, true);

  // create a scalar
  using ScalarType = cudf::scalar_type_t<T>;
  std::unique_ptr<cudf::scalar> s(new ScalarType{init, true});

  // calculate the expected value by CPU.
  thrust::host_vector<thrust::pair<T, bool>> value_and_validity(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 value_and_validity.begin(),
                 [](auto v, auto b) { return thrust::pair<T, bool>{v, b}; });

  // GPU test
  auto it_pair_dev = cudf::detail::make_pair_iterator<T>(*s);
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());
}
