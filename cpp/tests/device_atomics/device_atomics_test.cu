/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/bool.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cudf/detail/utilities/device_atomics.cuh>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/timestamp_utilities.cuh>
#include <tests/utilities/type_lists.hpp>

template <typename T>
__global__ void gpu_atomic_test(T* result, T* data, size_t size) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
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
__device__ T atomic_op(T* addr, T const& value, BinaryOp op) {
  T old_value = *addr;
  T assumed;

  do {
    assumed = old_value;
    T new_value = op(old_value, value);

    old_value = atomicCAS(addr, assumed, new_value);
  } while (assumed != old_value);

  return old_value;
}

template <typename T>
__global__ void gpu_atomicCAS_test(T* result, T* data, size_t size) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
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
typename std::enable_if_t<!cudf::is_timestamp<T>(), T> accumulate(
    std::vector<T> const& xs) {
  return std::accumulate(xs.begin(), xs.end(), T{0});
}

template <typename T>
typename std::enable_if_t<cudf::is_timestamp<T>(), T> accumulate(
    std::vector<T> const& xs) {
  auto ys = std::vector<typename T::rep>(xs.size());
  std::transform(xs.begin(), xs.end(), ys.begin(),
                 [](T const& ts) { return ts.time_since_epoch().count(); });
  return T{std::accumulate(ys.begin(), ys.end(), 0)};
}

template <typename T>
struct AtomicsTest : public cudf::test::BaseFixture {
  void atomic_test(std::vector<int> const& v_input,
                   bool is_cas_test,
                   int block_size = 0,
                   int grid_size = 1) {
    size_t vec_size = v_input.size();

    // use transform from std::vector<int> instead.
    std::vector<T> v(vec_size);
    std::transform(v_input.begin(), v_input.end(), v.begin(), [](int x) {
      T t(x);
      return t;
    });

    T exact[3];
    exact[0] = accumulate<T>(v);
    exact[1] = *(std::min_element(v.begin(), v.end()));
    exact[2] = *(std::max_element(v.begin(), v.end()));

    std::vector<T> result_init(6);
    result_init[0] = T{0};
    result_init[1] = std::numeric_limits<T>::max();
    result_init[2] = std::numeric_limits<T>::min();
    result_init[3] = result_init[0];
    result_init[4] = result_init[1];
    result_init[5] = result_init[2];

    thrust::device_vector<T> dev_data(v);
    thrust::device_vector<T> dev_result(result_init);

    if (block_size == 0) {
      block_size = vec_size;
    }

    if (is_cas_test) {
      gpu_atomicCAS_test<<<grid_size, block_size>>>(
          dev_result.data().get(), dev_data.data().get(), vec_size);
    } else {
      gpu_atomic_test<<<grid_size, block_size>>>(
          dev_result.data().get(), dev_data.data().get(), vec_size);
    }

    thrust::host_vector<T> host_result(dev_result);
    cudaDeviceSynchronize();
    CHECK_CUDA(0);

    EXPECT_EQ(host_result[0], exact[0]) << "atomicAdd test failed";
    EXPECT_EQ(host_result[1], exact[1]) << "atomicMin test failed";
    EXPECT_EQ(host_result[2], exact[2]) << "atomicMax test failed";
    EXPECT_EQ(host_result[3], exact[0]) << "atomicAdd test(2) failed";
    EXPECT_EQ(host_result[4], exact[1]) << "atomicMin test(2) failed";
    EXPECT_EQ(host_result[5], exact[2]) << "atomicMax test(2) failed";
  }
};

TYPED_TEST_CASE(AtomicsTest, cudf::test::FixedWidthTypes);

// tests for atomicAdd/Min/Max
TYPED_TEST(AtomicsTest, atomicOps) {
  bool is_cas_test = false;
  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test);
}

// tests for atomicCAS
TYPED_TEST(AtomicsTest, atomicCAS) {
  bool is_cas_test = true;
  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test);
}

// tests for atomicAdd/Min/Max
TYPED_TEST(AtomicsTest, atomicOpsGrid) {
  bool is_cas_test = false;
  int block_size = 3;
  int grid_size = 4;

  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test, block_size, grid_size);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test, block_size, grid_size);
}

// tests for atomicCAS
TYPED_TEST(AtomicsTest, atomicCASGrid) {
  bool is_cas_test = true;
  int block_size = 3;
  int grid_size = 4;

  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test, block_size, grid_size);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test, block_size, grid_size);
}

// tests for large array
TYPED_TEST(AtomicsTest, atomicOpsRandom) {
  bool is_cas_test = false;
  int block_size = 256;
  int grid_size = 64;

  std::vector<int> input_array(grid_size * block_size);

  std::default_random_engine engine;
  std::uniform_int_distribution<> dist(-10, 10);
  std::generate(input_array.begin(), input_array.end(),
                [&]() { return dist(engine); });

  this->atomic_test(input_array, is_cas_test, block_size, grid_size);
}

TYPED_TEST(AtomicsTest, atomicCASRandom) {
  bool is_cas_test = true;
  int block_size = 256;
  int grid_size = 64;

  std::vector<int> input_array(grid_size * block_size);

  std::default_random_engine engine;
  std::uniform_int_distribution<> dist(-10, 10);
  std::generate(input_array.begin(), input_array.end(),
                [&]() { return dist(engine); });

  this->atomic_test(input_array, is_cas_test, block_size, grid_size);
}

template <typename T>
__global__ void gpu_atomic_bitwiseOp_test(T* result, T* data, size_t size) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
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
  void atomic_test(std::vector<uint64_t> const& v_input,
                   int block_size = 0,
                   int grid_size = 1) {
    size_t vec_size = v_input.size();
    std::vector<T> v(vec_size);
    std::transform(v_input.begin(), v_input.end(), v.begin(), [](int x) {
      T t(x);
      return t;
    });

    std::vector<T> identity = {T(~0ull), T(0), T(0), T(~0ull), T(0), T(0)};
    T exact[3];
    exact[0] = std::accumulate(v.begin(), v.end(), identity[0],
                               [](T acc, uint64_t i) { return acc & T(i); });
    exact[1] = std::accumulate(v.begin(), v.end(), identity[1],
                               [](T acc, uint64_t i) { return acc | T(i); });
    exact[2] = std::accumulate(v.begin(), v.end(), identity[2],
                               [](T acc, uint64_t i) { return acc ^ T(i); });

    thrust::device_vector<T> dev_result(identity);
    thrust::device_vector<T> dev_data(v);

    if (block_size == 0) {
      block_size = vec_size;
    }

    gpu_atomic_bitwiseOp_test<T><<<grid_size, block_size>>>(
        reinterpret_cast<T*>(dev_result.data().get()),
        reinterpret_cast<T*>(dev_data.data().get()), vec_size);

    thrust::host_vector<T> host_result(dev_result);
    cudaDeviceSynchronize();
    CHECK_CUDA(0);

    print_exact(exact, "exact");
    print_exact(host_result.data(), "result");

    EXPECT_EQ(host_result[0], exact[0]) << "atomicAnd test failed";
    EXPECT_EQ(host_result[1], exact[1]) << "atomicOr  test failed";
    EXPECT_EQ(host_result[2], exact[2]) << "atomicXor test failed";
    EXPECT_EQ(host_result[3], exact[0]) << "atomicAnd test(2) failed";
    EXPECT_EQ(host_result[4], exact[1]) << "atomicOr  test(2) failed";
    EXPECT_EQ(host_result[5], exact[2]) << "atomicXor test(2) failed";
  }

  void print_exact(const T* v, const char* msg) {
    std::cout << std::hex << std::showbase;
    std::cout << "The " << msg << " = {" << +v[0] << ", " << +v[1] << ", "
              << +v[2] << "}" << std::endl;
  }
};

using BitwiseOpTestingTypes = cudf::test::Types<int8_t,
                                                int16_t,
                                                int32_t,
                                                int64_t,
                                                uint8_t,
                                                uint16_t,
                                                uint32_t,
                                                uint64_t>;

TYPED_TEST_CASE(AtomicsBitwiseOpTest, BitwiseOpTestingTypes);

TYPED_TEST(AtomicsBitwiseOpTest, atomicBitwiseOps) {
  {  // test for AND, XOR
    std::vector<uint64_t> input_array({0xfcfcfcfcfcfcfc7f, 0x7f7f7f7f7f7ffc,
                                       0xfffddffddffddfdf, 0x7f7f7f7f7f7ffc});
    this->atomic_test(input_array);
  }
  {  // test for OR, XOR
    std::vector<uint64_t> input_array({0x01, 0xfc02, 0x1dff03,
                                       0x1100a0b0801d0003, 0x8000000000000000,
                                       0x1dff03});
    this->atomic_test(input_array);
  }
}
