/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/timestamp_utilities.cuh>
#include <cudf_test/type_lists.hpp>

#include <thrust/host_vector.h>

#include <algorithm>

template <typename T>
__global__ void gpu_atomic_test(T* result, T* data, size_t size)
{
  size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  for (; id < size; id += step) {
    atomicAdd(&result[0], data[id]);
    atomicMin(&result[1], data[id]);
    atomicMax(&result[2], data[id]);
    cudf::genericAtomicOperation(&result[3], data[id], cudf::DeviceSum{});
    cudf::genericAtomicOperation(&result[4], data[id], cudf::DeviceMin{});
    cudf::genericAtomicOperation(&result[5], data[id], cudf::DeviceMax{});
  }
}

template <typename T, typename BinaryOp>
constexpr inline bool is_timestamp_sum()
{
  return cudf::is_timestamp<T>() && std::is_same_v<BinaryOp, cudf::DeviceSum>;
}
// Disable SUM of TIMESTAMP types
template <typename T,
          typename BinaryOp,
          std::enable_if_t<is_timestamp_sum<T, BinaryOp>()>* = nullptr>
__device__ T atomic_op(T* addr, T const& value, BinaryOp op)
{
  return {};
}

template <typename T,
          typename BinaryOp,
          std::enable_if_t<!is_timestamp_sum<T, BinaryOp>()>* = nullptr>
__device__ T atomic_op(T* addr, T const& value, BinaryOp op)
{
  T old_value = *addr;
  T assumed;

  do {
    assumed     = old_value;
    T new_value = op(old_value, value);

    old_value = atomicCAS(addr, assumed, new_value);
  } while (assumed != old_value);

  return old_value;
}

template <typename T>
__global__ void gpu_atomicCAS_test(T* result, T* data, size_t size)
{
  size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  for (; id < size; id += step) {
    atomic_op(&result[0], data[id], cudf::DeviceSum{});
    atomic_op(&result[1], data[id], cudf::DeviceMin{});
    atomic_op(&result[2], data[id], cudf::DeviceMax{});
    atomic_op(&result[3], data[id], cudf::DeviceSum{});
    atomic_op(&result[4], data[id], cudf::DeviceMin{});
    atomic_op(&result[5], data[id], cudf::DeviceMax{});
  }
}

template <typename T>
std::enable_if_t<!cudf::is_timestamp<T>(), T> accumulate(cudf::host_span<T const> xs)
{
  return std::accumulate(xs.begin(), xs.end(), T{0});
}

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), T> accumulate(cudf::host_span<T const> xs)
{
  auto ys = std::vector<typename T::rep>(xs.size());
  std::transform(
    xs.begin(), xs.end(), ys.begin(), [](T const& ts) { return ts.time_since_epoch().count(); });
  return T{typename T::duration{std::accumulate(ys.begin(), ys.end(), 0)}};
}

template <typename T>
struct AtomicsTest : public cudf::test::BaseFixture {
  void atomic_test(std::vector<int> const& v_input,
                   bool is_cas_test,
                   int block_size = 0,
                   int grid_size  = 1)
  {
    size_t vec_size = v_input.size();

    // use transform from thrust::host_vector<int> instead.
    thrust::host_vector<T> v(vec_size);
    std::transform(v_input.begin(), v_input.end(), v.begin(), [](int x) {
      T t = cudf::test::make_type_param_scalar<T>(x);
      return t;
    });

    T exact[3];
    exact[0] = accumulate<T>(v);
    exact[1] = *(std::min_element(v.begin(), v.end()));
    exact[2] = *(std::max_element(v.begin(), v.end()));

    thrust::host_vector<T> result_init(9);  // +3 padding for int8 tests
    result_init[0] = cudf::test::make_type_param_scalar<T>(0);
    if constexpr (cudf::is_chrono<T>()) {
      result_init[1] = T::max();
      result_init[2] = T::min();
    } else {
      result_init[1] = std::numeric_limits<T>::max();
      result_init[2] = std::numeric_limits<T>::min();
    }
    result_init[3] = result_init[0];
    result_init[4] = result_init[1];
    result_init[5] = result_init[2];

    auto dev_data   = cudf::detail::make_device_uvector_sync(v);
    auto dev_result = cudf::detail::make_device_uvector_sync(result_init);

    if (block_size == 0) { block_size = vec_size; }

    if (is_cas_test) {
      gpu_atomicCAS_test<<<grid_size, block_size, 0, cudf::get_default_stream().value()>>>(
        dev_result.data(), dev_data.data(), vec_size);
    } else {
      gpu_atomic_test<<<grid_size, block_size, 0, cudf::get_default_stream().value()>>>(
        dev_result.data(), dev_data.data(), vec_size);
    }

    auto host_result = cudf::detail::make_host_vector_sync(dev_result);

    CUDF_CHECK_CUDA(cudf::get_default_stream().value());

    if (!is_timestamp_sum<T, cudf::DeviceSum>()) {
      EXPECT_EQ(host_result[0], exact[0]) << "atomicAdd test failed";
    }
    EXPECT_EQ(host_result[1], exact[1]) << "atomicMin test failed";
    EXPECT_EQ(host_result[2], exact[2]) << "atomicMax test failed";
    if (!is_timestamp_sum<T, cudf::DeviceSum>()) {
      EXPECT_EQ(host_result[3], exact[0]) << "atomicAdd test(2) failed";
    }
    EXPECT_EQ(host_result[4], exact[1]) << "atomicMin test(2) failed";
    EXPECT_EQ(host_result[5], exact[2]) << "atomicMax test(2) failed";
  }
};

TYPED_TEST_SUITE(AtomicsTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

// tests for atomicAdd/Min/Max
TYPED_TEST(AtomicsTest, atomicOps)
{
  bool is_cas_test = false;
  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test);
}

// tests for atomicCAS
TYPED_TEST(AtomicsTest, atomicCAS)
{
  bool is_cas_test = true;
  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test);
}

// tests for atomicAdd/Min/Max
TYPED_TEST(AtomicsTest, atomicOpsGrid)
{
  bool is_cas_test = false;
  int block_size   = 3;
  int grid_size    = 4;

  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test, block_size, grid_size);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test, block_size, grid_size);
}

// tests for atomicCAS
TYPED_TEST(AtomicsTest, atomicCASGrid)
{
  bool is_cas_test = true;
  int block_size   = 3;
  int grid_size    = 4;

  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test, block_size, grid_size);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test, block_size, grid_size);
}

// tests for large array
TYPED_TEST(AtomicsTest, atomicOpsRandom)
{
  bool is_cas_test = false;
  int block_size   = 256;
  int grid_size    = 64;

  std::vector<int> input_array(grid_size * block_size);

  std::default_random_engine engine;
  std::uniform_int_distribution<> dist(-10, 10);
  std::generate(input_array.begin(), input_array.end(), [&]() { return dist(engine); });

  this->atomic_test(input_array, is_cas_test, block_size, grid_size);
}

TYPED_TEST(AtomicsTest, atomicCASRandom)
{
  bool is_cas_test = true;
  int block_size   = 256;
  int grid_size    = 64;

  std::vector<int> input_array(grid_size * block_size);

  std::default_random_engine engine;
  std::uniform_int_distribution<> dist(-10, 10);
  std::generate(input_array.begin(), input_array.end(), [&]() { return dist(engine); });

  this->atomic_test(input_array, is_cas_test, block_size, grid_size);
}

template <typename T>
__global__ void gpu_atomic_bitwiseOp_test(T* result, T* data, size_t size)
{
  size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  for (; id < size; id += step) {
    atomicAnd(&result[0], data[id]);
    atomicOr(&result[1], data[id]);
    atomicXor(&result[2], data[id]);
    cudf::genericAtomicOperation(&result[3], data[id], cudf::DeviceAnd{});
    cudf::genericAtomicOperation(&result[4], data[id], cudf::DeviceOr{});
    cudf::genericAtomicOperation(&result[5], data[id], cudf::DeviceXor{});
  }
}

template <typename T>
struct AtomicsBitwiseOpTest : public cudf::test::BaseFixture {
  void atomic_test(std::vector<uint64_t> const& v_input, int block_size = 0, int grid_size = 1)
  {
    size_t vec_size = v_input.size();
    std::vector<T> v(vec_size);
    std::transform(v_input.begin(), v_input.end(), v.begin(), [](int x) {
      T t(x);
      return t;
    });

    thrust::host_vector<T> identity(9, T{0});  // +3 elements padding for int8 tests
    identity[0] = T(~0ull);
    identity[3] = T(~0ull);

    T exact[3];
    exact[0] = std::accumulate(
      v.begin(), v.end(), identity[0], [](T acc, uint64_t i) { return acc & T(i); });
    exact[1] = std::accumulate(
      v.begin(), v.end(), identity[1], [](T acc, uint64_t i) { return acc | T(i); });
    exact[2] = std::accumulate(
      v.begin(), v.end(), identity[2], [](T acc, uint64_t i) { return acc ^ T(i); });

    auto dev_result = cudf::detail::make_device_uvector_sync(identity);
    auto dev_data   = cudf::detail::make_device_uvector_sync(v);

    if (block_size == 0) { block_size = vec_size; }

    gpu_atomic_bitwiseOp_test<T><<<grid_size, block_size, 0, cudf::get_default_stream().value()>>>(
      reinterpret_cast<T*>(dev_result.data()), reinterpret_cast<T*>(dev_data.data()), vec_size);

    auto host_result = cudf::detail::make_host_vector_sync(dev_result);

    CUDF_CHECK_CUDA(cudf::get_default_stream().value());

    // print_exact(exact, "exact");
    // print_exact(host_result.data(), "result");

    EXPECT_EQ(host_result[0], exact[0]) << "atomicAnd test failed";
    EXPECT_EQ(host_result[1], exact[1]) << "atomicOr  test failed";
    EXPECT_EQ(host_result[2], exact[2]) << "atomicXor test failed";
    EXPECT_EQ(host_result[3], exact[0]) << "atomicAnd test(2) failed";
    EXPECT_EQ(host_result[4], exact[1]) << "atomicOr  test(2) failed";
    EXPECT_EQ(host_result[5], exact[2]) << "atomicXor test(2) failed";
  }

  [[maybe_unused]] void print_exact(const T* v, const char* msg)
  {
    std::cout << std::hex << std::showbase;
    std::cout << "The " << msg << " = {" << +v[0] << ", " << +v[1] << ", " << +v[2] << "}"
              << std::endl;
  }
};

using BitwiseOpTestingTypes =
  cudf::test::Types<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>;

TYPED_TEST_SUITE(AtomicsBitwiseOpTest, BitwiseOpTestingTypes);

TYPED_TEST(AtomicsBitwiseOpTest, atomicBitwiseOps)
{
  {  // test for AND, XOR
    std::vector<uint64_t> input_array(
      {0xfcfc'fcfc'fcfc'fc7f, 0x7f'7f7f'7f7f'7ffc, 0xfffd'dffd'dffd'dfdf, 0x7f'7f7f'7f7f'7ffc});
    this->atomic_test(input_array);
  }
  {  // test for OR, XOR
    std::vector<uint64_t> input_array(
      {0x01, 0xfc02, 0x1d'ff03, 0x1100'a0b0'801d'0003, 0x8000'0000'0000'0000, 0x1d'ff03});
    this->atomic_test(input_array);
  }
}

CUDF_TEST_PROGRAM_MAIN()
