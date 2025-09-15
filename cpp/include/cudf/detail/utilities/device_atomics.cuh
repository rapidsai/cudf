/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

/**
 * @brief overloads for CUDA atomic operations
 * @file device_atomics.cuh
 *
 * Provides the overloads for all of cudf's data types, specifically int8_t,
 * int16_t, int32_t, int64_t, float, double, cudf::timestamp_D,
 * cudf::timestamp_s, cudf::timestamp_ms, cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool for the CUDA atomic operations
 * `atomicAdd`, `atomicMin`, `atomicMax`, `atomicCAS`.
 *
 * Also provides `cudf::detail::genericAtomicOperation` which performs an
 * atomic operation with the given binary operator.
 */

#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/atomic>
#include <cuda/std/type_traits>

namespace cudf {
namespace detail {

template <typename T, typename Op>
struct genericAtomicOperationImpl;

template <typename T, typename Op>
struct genericAtomicOperationImpl {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, Op op)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T old_value = atomic_addr.load(cuda::memory_order_relaxed);
    T new_value;

    do {
      new_value = op(old_value, update_value);
    } while (!atomic_addr.compare_exchange_weak(old_value, new_value, cuda::memory_order_relaxed));

    return old_value;
  }
};

// Optimized specializations using native atomic operations
template <typename T>
struct genericAtomicOperationImpl<T, DeviceSum> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceSum op)
    requires(cuda::std::is_arithmetic_v<T> && !cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    return atomic_addr.fetch_add(update_value, cuda::memory_order_relaxed);
  }

  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceSum op)
    requires(!cuda::std::is_arithmetic_v<T> || cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T old_value = atomic_addr.load(cuda::memory_order_relaxed);
    T new_value;

    do {
      new_value = op(old_value, update_value);
    } while (!atomic_addr.compare_exchange_weak(old_value, new_value, cuda::memory_order_relaxed));

    return old_value;
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, DeviceMin> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMin op)
    requires(cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    return atomic_addr.fetch_min(update_value, cuda::memory_order_relaxed);
  }

  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMin op)
    requires(!cuda::std::is_integral_v<T> || cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T old_value = atomic_addr.load(cuda::memory_order_relaxed);
    T new_value;

    do {
      new_value = op(old_value, update_value);
    } while (!atomic_addr.compare_exchange_weak(old_value, new_value, cuda::memory_order_relaxed));

    return old_value;
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, DeviceMax> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMax op)
    requires(cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    return atomic_addr.fetch_max(update_value, cuda::memory_order_relaxed);
  }

  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMax op)
    requires(!cuda::std::is_integral_v<T> || cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T old_value = atomic_addr.load(cuda::memory_order_relaxed);
    T new_value;

    do {
      new_value = op(old_value, update_value);
    } while (!atomic_addr.compare_exchange_weak(old_value, new_value, cuda::memory_order_relaxed));

    return old_value;
  }
};

// -----------------------------------------------------------------------
// the implementation of `typesAtomicCASImpl`
template <typename T>
struct typesAtomicCASImpl {
  __forceinline__ __device__ T operator()(T* addr, T const& compare, T const& update_value)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T expected = compare;
    atomic_addr.compare_exchange_strong(expected, update_value, cuda::memory_order_relaxed);
    return expected;
  }
};

/**
 * @brief Compute atomic binary operation
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes 'BinaryOp'('old', 'update_value'),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `genericAtomicOperation` are:
 * int8_t, int16_t, int32_t, int64_t, float, double
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be computed
 * @param op  The binary operator used for compute
 *
 * @returns The old value at `address`
 */
template <typename T, typename BinaryOp>
T __forceinline__ __device__ genericAtomicOperation(T* address, T const& update_value, BinaryOp op)
  requires(cudf::is_numeric<T>())
{
  auto fun = cudf::detail::genericAtomicOperationImpl<T, BinaryOp>{};
  return T(fun(address, update_value, op));
}

// specialization for cudf::detail::timestamp types
template <typename T, typename BinaryOp>
T __forceinline__ __device__ genericAtomicOperation(T* address, T const& update_value, BinaryOp op)
  requires(cudf::is_timestamp<T>())
{
  using R = typename T::rep;
  // Unwrap the input timestamp to its underlying duration value representation.
  // Use the underlying representation's type to apply operation for the cudf::detail::timestamp
  auto update_value_rep = update_value.time_since_epoch().count();
  auto fun              = cudf::detail::genericAtomicOperationImpl<R, BinaryOp>{};
  return T{T::duration(fun(reinterpret_cast<R*>(address), update_value_rep, op))};
}

// specialization for cudf::detail::duration types
template <typename T, typename BinaryOp>
T __forceinline__ __device__ genericAtomicOperation(T* address, T const& update_value, BinaryOp op)
  requires(cudf::is_duration<T>())
{
  using R = typename T::rep;
  // Unwrap the input duration to its underlying duration value representation.
  // Use the underlying representation's type to apply operation for the cudf::detail::duration
  auto update_value_rep = update_value.count();
  auto fun              = cudf::detail::genericAtomicOperationImpl<R, BinaryOp>{};
  return T(fun(reinterpret_cast<R*>(address), update_value_rep, op));
}

// specialization for bool types
template <typename BinaryOp>
__forceinline__ __device__ bool genericAtomicOperation(bool* address,
                                                       bool const& update_value,
                                                       BinaryOp op)
{
  using T = bool;
  // don't use underlying type to apply operation for bool
  auto fun = cudf::detail::genericAtomicOperationImpl<T, BinaryOp>{};
  return T(fun(address, update_value, op));
}

/**
 * @brief Overloads for `atomic_add`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes (old + val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomic_add` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::timestamp_D, cudf::timestamp_s, cudf::timestamp_ms cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool
 *
 * CUDA natively supports `int32_t`, `uint32_t`, `uint64_t`, `float`, `double.
 * (`double` is supported after Pascal).
 * Other types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be added
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_add(T* address, T val)
{
  return cudf::detail::genericAtomicOperation(address, val, cudf::DeviceSum{});
}

/**
 * @brief Overloads for `atomic_mul`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes (old * val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomicMul` are:
 * int8_t, int16_t, int32_t, int64_t, float, double, and bool
 *
 * All types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be multiplied
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_mul(T* address, T val)
{
  return cudf::detail::genericAtomicOperation(address, val, cudf::DeviceProduct{});
}

/**
 * @brief Overloads for `atomic_min`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes the minimum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomic_min` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::timestamp_D, cudf::timestamp_s, cudf::timestamp_ms, cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool
 *
 * CUDA natively supports `int32_t`, `uint32_t`, `int64_t`, `uint64_t`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_min(T* address, T val)
{
  return cudf::detail::genericAtomicOperation(address, val, cudf::DeviceMin{});
}

/**
 * @brief Overloads for `atomic_max`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes the maximum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomic_max` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::timestamp_D, cudf::timestamp_s, cudf::timestamp_ms, cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool
 *
 * CUDA natively supports `int32_t`, `uint32_t`, `int64_t`, `uint64_t`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_max(T* address, T val)
{
  return cudf::detail::genericAtomicOperation(address, val, cudf::DeviceMax{});
}

/**
 * @brief Overloads for `atomic_cas`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes (`old` == `compare` ? `val` : `old`),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomic_cas` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::timestamp_D, cudf::timestamp_s, cudf::timestamp_ms, cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool
 * CUDA natively supports `int32_t`, `uint32_t`, `uint64_t`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param compare The value to be compared
 * @param val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_cas(T* address, T compare, T val)
{
  return cudf::detail::typesAtomicCASImpl<T>()(address, compare, val);
}

}  // namespace detail
}  // namespace cudf
