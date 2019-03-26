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

/** ---------------------------------------------------------------------------*
 * @brief overloads for CUDA atomic operations
 * @file device_atomics.cuh
 *
 * Provides the overloads for all of possible cudf's data types,
 * where cudf's data types are, int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category.
 * where CUDA atomic operations are, `atomicAdd`, `atomicMin`, `atomicMax`,
 * `atomicCAS` (* see note).
 *
 * @note atomicCAS doesn't provides overloads for int8_t, int16_t
 * ---------------------------------------------------------------------------**/



#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/wrapper_types.hpp"
#include "utilities/error_utils.hpp"

namespace cudf {

namespace detail {

    template <typename T_output, typename T_input>
    __forceinline__  __device__
    T_output type_reinterpret(T_input value)
    {
        return *( reinterpret_cast<T_output*>(&value) );
    }

    // the implementation of `genericAtomicOperation`
    template <typename T, typename Op, size_t n>
    struct genericAtomicOperationImpl;

    // single byte atomic operation
    template<typename T, typename Op>
    struct genericAtomicOperationImpl<T, Op, 1> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, Op op)
        {
            using T_int = unsigned int;

            T_int * address_uint32 = reinterpret_cast<T_int *>
                (addr - (reinterpret_cast<size_t>(addr) & 3));
            unsigned int shift = ((reinterpret_cast<size_t>(addr) & 3) * 8);

            T_int old = *address_uint32;
            T_int assumed ;

            do {
                assumed = old;
                T target_value = T((old >> shift) & 0xff);
                uint8_t new_value = type_reinterpret<uint8_t, T>
                    ( op(target_value, update_value) );
                old = (old & ~(0x000000ff << shift)) | (T_int(new_value) << shift);
                old = atomicCAS(address_uint32, assumed, old);
            } while (assumed != old);

            return T((old >> shift) & 0xff);
        }
    };

    // 2 bytes atomic operation
    template<typename T, typename Op>
    struct genericAtomicOperationImpl<T, Op, 2> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, Op op)
        {
            using T_int = unsigned int;
            bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
            T_int * address_uint32 = reinterpret_cast<T_int *>
                (reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

            T_int old = *address_uint32;
            T_int assumed ;

            do {
                assumed = old;
                T target_value = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
                uint16_t new_value = type_reinterpret<uint16_t, T>
                    ( op(target_value, update_value) );

                old = (is_32_align) ? (old & 0xffff0000) | new_value
                                    : (old & 0xffff) | (T_int(new_value) << 16);
                old = atomicCAS(address_uint32, assumed, old);
            } while (assumed != old);

            return (is_32_align) ? T(old & 0xffff) : T(old >> 16);;
        }
    };

    // 4 bytes atomic operation
    template<typename T, typename Op>
    struct genericAtomicOperationImpl<T, Op, 4> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, Op op)
        {
            using T_int = unsigned int;

            T old_value = *addr;
            T assumed {old_value};

            do {
                assumed  = old_value;
                const T new_value = op(old_value, update_value);

                T_int ret = atomicCAS(
                    reinterpret_cast<T_int*>(addr),
                    type_reinterpret<T_int, T>(assumed),
                    type_reinterpret<T_int, T>(new_value));
                old_value = type_reinterpret<T, T_int>(ret);

            } while (assumed != old_value);

            return old_value;
        }
    };

    // 8 bytes atomic operation
    template<typename T, typename Op>
    struct genericAtomicOperationImpl<T, Op, 8> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, Op op)
        {
            using T_int = unsigned long long int;

            T old_value = *addr;
            T assumed {old_value};

            do {
                assumed  = old_value;
                const T new_value = op(old_value, update_value);

                T_int ret = atomicCAS(
                    reinterpret_cast<T_int*>(addr),
                    type_reinterpret<T_int, T>(assumed),
                    type_reinterpret<T_int, T>(new_value));
                old_value = type_reinterpret<T, T_int>(ret);

            } while (assumed != old_value);

            return old_value;
        }
    };

    // the implementation of `typesAtomicCASImpl`
    template <typename T, size_t n>
    struct typesAtomicCASImpl;

    template<typename T>
    struct typesAtomicCASImpl<T, 4> {
        __forceinline__  __device__
        T operator()(T* addr, T const & compare, T const & update_value)
        {
            using T_int = unsigned int;

            T_int ret = atomicCAS(
                reinterpret_cast<T_int*>(addr),
                type_reinterpret<T_int, T>(compare),
                type_reinterpret<T_int, T>(update_value));

            return type_reinterpret<T, T_int>(ret);
        }
    };

    // 8 bytes atomic operation
    template<typename T>
    struct typesAtomicCASImpl<T, 8> {
        __forceinline__  __device__
        T operator()(T* addr, T const & compare, T const & update_value)
        {
            using T_int = unsigned long long int;

            T_int ret = atomicCAS(
                reinterpret_cast<T_int*>(addr),
                type_reinterpret<T_int, T>(compare),
                type_reinterpret<T_int, T>(update_value));

            return type_reinterpret<T, T_int>(ret);
        }
    };

    // call atomic function with type cast between same underlying type
    template <typename T, typename Functor>
    __forceinline__  __device__
    T typesAtomicOperation32(T* addr, T val, Functor atomicFunc)
    {
        using T_int = int;
        T_int ret = atomicFunc(reinterpret_cast<T_int*>(addr),
            cudf::detail::type_reinterpret<T_int, T>(val));

        return cudf::detail::type_reinterpret<T, T_int>(ret);
    }

    // call atomic function with type cast between same underlying type
    template <typename T, typename Functor>
    __forceinline__  __device__
    T typesAtomicOperation64(T* addr, T val, Functor atomicFunc)
    {
        using T_int = long long int;
        T_int ret = atomicFunc(reinterpret_cast<T_int*>(addr),
            cudf::detail::type_reinterpret<T_int, T>(val));

        return cudf::detail::type_reinterpret<T, T_int>(ret);
    }

} // namespace detail


template <typename T, typename BinaryOp>
__forceinline__  __device__
T genericAtomicOperation(T* address, T const & update_value, BinaryOp op)
{
    return cudf::detail::genericAtomicOperationImpl<T, BinaryOp, sizeof(T)>()
        (address, update_value, op);
}

// ------------------------------------------------------------------------
// Binary ops for sum, min, max
struct DeviceSum {
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs + rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

struct DeviceMin{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs <= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
};

struct DeviceMax{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs >= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::lowest(); }
};

} // namespace cudf



/* Overloads for `atomicAdd` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes (old + val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomicAdd` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category.
 * Cuda natively supports `sint32`, `uint32`, `uint64`, `float`, `double`
 * (`double` is supported after Pascal).
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be added
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int8_t atomicAdd(int8_t* address, int8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceSum{});
}

/**
 * @overload int16_t atomicAdd(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicAdd(int16_t* address, int16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceSum{});
}

/**
 * @overload int64_t atomicAdd(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicAdd(int64_t* address, int64_t val)
{
    // `atomicAdd` supports uint64_t, but not int64_t
    return cudf::genericAtomicOperation(address, val, cudf::DeviceSum{});
}

#if defined(__CUDA_ARCH__) && ( __CUDA_ARCH__ < 600 )
/**
 * @overload double atomicAdd(double* address, double val)
 */
__forceinline__ __device__
double atomicAdd(double* address, double val)
{
    // `atomicAdd` for `double` is supported from Pascal
    return cudf::genericAtomicOperation(address, val, cudf::DeviceSum{});
}
#endif

/**
 * @overload cudf::date32 atomicAdd(cudf::date32* address, cudf::date32 val)
 */
inline  __device__
cudf::date32 atomicAdd(cudf::date32* address, cudf::date32 val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
        (address, val, [](T* a, T v){return atomicAdd(a, v);});
}

/**
 * @overload cudf::category atomicAdd(cudf::category* address, cudf::category val)
 */
__forceinline__ __device__
cudf::category atomicAdd(cudf::category* address, cudf::category val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
        (address, val, [](T* a, T v){return atomicAdd(a, v);});
}

/**
 * @overload cudf::date64 atomicAdd(cudf::date64* address, cudf::date64 val)
 */
__forceinline__ __device__
cudf::date64 atomicAdd(cudf::date64* address, cudf::date64 val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceSum{});
}

/**
 * @overload cudf::timestamp atomicAdd(cudf::timestamp* address, cudf::timestamp val)
 */
__forceinline__ __device__
cudf::timestamp atomicAdd(cudf::timestamp* address, cudf::timestamp val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceSum{});
}

/* Overloads for `atomicMin` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes the minimum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomicMin` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category.
 * Cuda natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int8_t atomicMin(int8_t* address, int8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMin{});
}

/**
 * @overload int16_t atomicMin(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicMin(int16_t* address, int16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMin{});
}

/**
 * @overload int64_t atomicMin(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicMin(int64_t* address, int64_t val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicMin(a, v);});
}

/**
 * @overload float atomicMin(float* address, float val)
 */
__forceinline__ __device__
float atomicMin(float* address, float val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMin{});
}

/**
 * @overload double atomicMin(double* address, double val)
 */
__forceinline__ __device__
double atomicMin(double* address, double val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMin{});
}

/**
 * @overload cudf::date32 atomicMin(cudf::date32* address, cudf::date32 val)
 */
inline  __device__
cudf::date32 atomicMin(cudf::date32* address, cudf::date32 val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
        (address, val, [](T* a, T v){return atomicMin(a, v);});
}

/**
 * @overload cudf::category atomicMin(cudf::category* address, cudf::category val)
 */
__forceinline__ __device__
cudf::category atomicMin(cudf::category* address, cudf::category val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
        (address, val, [](T* a, T v){return atomicMin(a, v);});
}

/**
 * @overload cudf::date64 atomicMin(cudf::date64* address, cudf::date64 val)
 */
__forceinline__ __device__
cudf::date64 atomicMin(cudf::date64* address, cudf::date64 val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicMin(a, v);});
}

/**
 * @overload cudf::timestamp atomicMin(cudf::timestamp* address, cudf::timestamp val)
 */
__forceinline__ __device__
cudf::timestamp atomicMin(cudf::timestamp* address, cudf::timestamp val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicMin(a, v);});
}

/* Overloads for `atomicMax` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes the maximum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomicMax` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category.
 * Cuda natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int8_t atomicMax(int8_t* address, int8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMax{});
}

/**
 * @overload int16_t atomicMax(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicMax(int16_t* address, int16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMax{});
}

/**
 * @overload int64_t atomicMax(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicMax(int64_t* address, int64_t val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicMax(a, v);});
}

/**
 * @overload float atomicMax(float* address, float val)
 */
__forceinline__ __device__
float atomicMax(float* address, float val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMax{});
}

/**
 * @overload double atomicMax(double* address, double val)
 */
__forceinline__ __device__
double atomicMax(double* address, double val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMax{});
}

/**
 * @overload cudf::date32 atomicMax(cudf::date32* address, cudf::date32 val)
 */inline  __device__
cudf::date32 atomicMax(cudf::date32* address, cudf::date32 val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
        (address, val, [](T* a, T v){return atomicMax(a, v);});
}

/**
 * @overload cudf::category atomicMax(cudf::category* address, cudf::category val)
 */
__forceinline__ __device__
cudf::category atomicMax(cudf::category* address, cudf::category val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
        (address, val, [](T* a, T v){return atomicMax(a, v);});
}

/**
 * @overload cudf::date64 atomicMax(cudf::date64* address, cudf::date64 val)
 */
__forceinline__ __device__
cudf::date64 atomicMax(cudf::date64* address, cudf::date64 val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicMax(a, v);});
}

/**
 * @overload cudf::timestamp atomicMax(cudf::timestamp* address, cudf::timestamp val)
 */
__forceinline__ __device__
cudf::timestamp atomicMax(cudf::timestamp* address, cudf::timestamp val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicMax(a, v);});
}

/* Overloads for `atomicCAS` */
/** --------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes the maximum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomicCAS` are:
 * int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category.
 * int8_t, int16_t are not supported as overloads
 * Cuda natively supports `sint32`, `uint32`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 *
 * @note int8_t, int16_t are not supported as `atomicCAS` overloads 
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val)
{
    using T = int64_t;
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload float atomicCAS(float* address, float compare, float val)
 */
__forceinline__ __device__
float atomicCAS(float* address, float compare, float val)
{
    using T = float;
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload double atomicCAS(double* address, double compare, double val)
 */
__forceinline__ __device__
double atomicCAS(double* address, double compare, double val)
{
    using T = double;
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload cudf::date32 atomicCAS(cudf::date32* address, cudf::date32 compare, cudf::date32 val)
 */
inline  __device__
cudf::date32 atomicCAS(cudf::date32* address, cudf::date32 compare, cudf::date32 val)
{
    using T = cudf::date32;
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload cudf::category atomicCAS(cudf::category* address, cudf::category compare, cudf::category val)
 */
__forceinline__ __device__
cudf::category atomicCAS(cudf::category* address, cudf::category compare, cudf::category val)
{
    using T = cudf::category;
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload cudf::date64 atomicCAS(cudf::date64* address, cudf::date64 compare, cudf::date64 val)
 */
__forceinline__ __device__
cudf::date64 atomicCAS(cudf::date64* address, cudf::date64 compare, cudf::date64 val)
{
    using T = cudf::date64;
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload cudf::timestamp atomicCAS(cudf::timestamp* address, cudf::timestamp compare, cudf::timestamp val)
 */
__forceinline__ __device__
cudf::timestamp atomicCAS(cudf::timestamp* address, cudf::timestamp compare, cudf::timestamp val)
{
    using T = cudf::timestamp;
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

