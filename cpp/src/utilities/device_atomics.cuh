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

#ifndef DEVICE_ATOMICS_CUH
#define DEVICE_ATOMICS_CUH

/** ---------------------------------------------------------------------------*
 * @brief overloads for CUDA atomic operations
 * @file device_atomics.cuh
 *
 * Provides the overloads for all of possible cudf's data types,
 * where cudf's data types are, int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category,
 * cudf::nvstring_category, cudf::bool8,
 * where CUDA atomic operations are, `atomicAdd`, `atomicMin`, `atomicMax`,
 * `atomicCAS`.
 * `atomicAnd`, `atomicOr`, `atomicXor` are also supported for integer data types.
 * Also provides `cudf::genericAtomicOperation` which performs atomic operation 
 * with the given binary operator.
 * ---------------------------------------------------------------------------**/

#include <cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/wrapper_types.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/device_operators.cuh>

namespace cudf {
namespace detail {

    template <typename T_output, typename T_input>
    __forceinline__  __device__
    T_output type_reinterpret(T_input value)
    {
        return *( reinterpret_cast<T_output*>(&value) );
    }

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


    // -----------------------------------------------------------------------
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
            T_int shift = ((reinterpret_cast<size_t>(addr) & 3) * 8);

            T_int old = *address_uint32;
            T_int assumed ;

            do {
                assumed = old;
                T target_value = T((old >> shift) & 0xff);
                uint8_t updating_value = type_reinterpret<uint8_t, T>
                    ( op(target_value, update_value) );
                T_int new_value = (old & ~(0x000000ff << shift))
                    | (T_int(updating_value) << shift);
                old = atomicCAS(address_uint32, assumed, new_value);
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
                uint16_t updating_value = type_reinterpret<uint16_t, T>
                    ( op(target_value, update_value) );

                T_int new_value  = (is_32_align)
                    ? (old & 0xffff0000) | updating_value
                    : (old & 0xffff) | (T_int(updating_value) << 16);
                old = atomicCAS(address_uint32, assumed, new_value);
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

    // -----------------------------------------------------------------------
    // specialized functions for operators
    // `atomicAdd` supports int32, float, double (signed int64 is not supproted.)
    // `atomicMin`, `atomicMax` support int32_t, int64_t
    template<>
    struct genericAtomicOperationImpl<float, DeviceSum, 4> {
        using T = float;
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceSum op)
        {
            return atomicAdd(addr, update_value);
        }
    };

#if defined(__CUDA_ARCH__) && ( __CUDA_ARCH__ >= 600 )
    template<>
    struct genericAtomicOperationImpl<double, DeviceSum, 8> {
        using T = double;
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceSum op)
        {
            return atomicAdd(addr, update_value);
        }
    };
#endif

    template<>
    struct genericAtomicOperationImpl<int32_t, DeviceSum, 4> {
        using T = int32_t;
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceSum op)
        {
            return atomicAdd(addr, update_value);
        }
    };

    template<>
    struct genericAtomicOperationImpl<int32_t, DeviceMin, 4> {
        using T = int32_t;
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceMin op)
        {
            return atomicMin(addr, update_value);
        }
    };

    template<>
    struct genericAtomicOperationImpl<int32_t, DeviceMax, 4> {
        using T = int32_t;
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceMax op)
        {
            return atomicMax(addr, update_value);
        }
    };

    template<>
    struct genericAtomicOperationImpl<int64_t, DeviceMin, 8> {
        using T = int64_t;
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceMin op)
        {
            using T_out = long long int;
            T ret = atomicMin(reinterpret_cast<T_out*>(addr),
                type_reinterpret<T_out, T>(update_value) );
            return ret;
        }
    };

    template<>
    struct genericAtomicOperationImpl<int64_t, DeviceMax, 8> {
        using T = int64_t;
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceMax op)
        {
            using T_out = long long int;
            T ret = atomicMax(reinterpret_cast<T_out*>(addr),
                type_reinterpret<T_out, T>(update_value) );
            return ret;
        }
    };

    template<typename T>
    struct genericAtomicOperationImpl<T, DeviceAnd, 4> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceAnd op)
        {
            return atomicAnd(addr, update_value);
        }
    };

    template<typename T>
    struct genericAtomicOperationImpl<T, DeviceAnd, 8> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceAnd op)
        {
            using T_int = long long int;
            return cudf::detail::typesAtomicOperation64
                (addr, update_value, [](T_int* a, T_int v){return atomicAnd(a, v);});
        }
    };

    template<typename T>
    struct genericAtomicOperationImpl<T, DeviceOr, 4> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceOr op)
        {
            return atomicOr(addr, update_value);
        }
    };

    template<typename T>
    struct genericAtomicOperationImpl<T, DeviceOr, 8> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceOr op)
        {
            using T_int = long long int;
            return cudf::detail::typesAtomicOperation64
                (addr, update_value, [](T_int* a, T_int v){return atomicOr(a, v);});
        }
    };

    template<typename T>
    struct genericAtomicOperationImpl<T, DeviceXor, 4> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceXor op)
        {
            return atomicXor(addr, update_value);
        }
    };

    template<typename T>
    struct genericAtomicOperationImpl<T, DeviceXor, 8> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, DeviceXor op)
        {
            using T_int = long long int;
            return cudf::detail::typesAtomicOperation64
                (addr, update_value, [](T_int* a, T_int v){return atomicXor(a, v);});
        }
    };

    // -----------------------------------------------------------------------
    // the implementation of `typesAtomicCASImpl`
    template <typename T, size_t n>
    struct typesAtomicCASImpl;

    template<typename T>
    struct typesAtomicCASImpl<T, 1> {
        __forceinline__  __device__
        T operator()(T* addr, T const & compare, T const & update_value)
        {
            using T_int = unsigned int;

            T_int shift = ((reinterpret_cast<size_t>(addr) & 3) * 8);
            T_int * address_uint32 = reinterpret_cast<T_int *>
                (addr - (reinterpret_cast<size_t>(addr) & 3));

            // the 'target_value' in `old` can be different from `compare`
            // because other thread may update the value
            // before fetching a value from `address_uint32` in this function
            T_int old = *address_uint32;
            T_int assumed;
            T target_value;
            uint8_t u_val = type_reinterpret<uint8_t, T>(update_value);

            do {
               assumed = old;
               target_value = T((old >> shift) & 0xff);
               // have to compare `target_value` and `compare` before calling atomicCAS
               // the `target_value` in `old` can be different with `compare`
               if( target_value != compare ) break;

               T_int new_value = (old & ~(0x000000ff << shift))
                   | (T_int(u_val) << shift);
               old = atomicCAS(address_uint32, assumed, new_value);
            } while (assumed != old);

            return target_value;
        }
    };

    template<typename T>
    struct typesAtomicCASImpl<T, 2> {
        __forceinline__  __device__
        T operator()(T* addr, T const & compare, T const & update_value)
        {
            using T_int = unsigned int;

            bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
            T_int * address_uint32 = reinterpret_cast<T_int *>
                (reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

            T_int old = *address_uint32;
            T_int assumed;
            T target_value;
            uint16_t u_val = type_reinterpret<uint16_t, T>(update_value);

            do {
                assumed = old;
                target_value = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
                if( target_value != compare ) break;

                T_int new_value = (is_32_align) ? (old & 0xffff0000) | u_val
                    : (old & 0xffff) | (T_int(u_val) << 16);
                old = atomicCAS(address_uint32, assumed, new_value);
            } while (assumed != old);

            return target_value;
        }
    };

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

} // namespace detail

/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory,
 * computes 'BinaryOp'('old', 'update_value'),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `genericAtomicOperation` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category,
 * cudf::nvstring_category, cudf::bool8
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be added
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T, typename BinaryOp>
__forceinline__  __device__
T genericAtomicOperation(T* address, T const & update_value, BinaryOp op)
{
    auto ret=  cudf::detail::genericAtomicOperationImpl<T, BinaryOp, sizeof(T)>{}
            (address, update_value, op);
    return T(ret);
}


template <typename T, gdf_dtype dtype, typename BinaryOp, typename W = cudf::detail::wrapper<T, dtype> >
W genericAtomicOperator( W* address, W const& update_value, BinaryOp op){
    // unwrap the input type to expect
    // that the native atomic API is used for the underlying type
    auto ret=  cudf::detail::genericAtomicOperationImpl<T, BinaryOp, sizeof(T)>{}
            (static_cast<T*>(address), cudf::detail::unwrap(update_value), op);
    return W(ret);
}


template <typename BinaryOp>
__forceinline__  __device__
cudf::bool8 genericAtomicOperation(cudf::bool8* address, cudf::bool8 const & update_value, BinaryOp op)
{
    using T = cudf::bool8;
    // don't use underlying type to apply operation for cudf::bool8
    auto ret = cudf::detail::genericAtomicOperationImpl<T, BinaryOp, sizeof(T)>()
            (address, update_value, op);

    return T(ret);
}

} // namespace cudf



/* Overloads for `atomicAdd` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory,
 * computes (old + val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomicAdd` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category,
 * cudf::nvstring_category, cudf::bool8
 * Cuda natively supports `sint32`, `uint32`, `uint64`, `float`, `double.
 * (`double` is supported after Pascal).
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be added
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T>
__forceinline__ __device__
T atomicAdd(T* address, T val)
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
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category,
 * cudf::nvstring_category
 * Cuda natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T>
__forceinline__ __device__
T atomicMin(T* address, T val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMin{});
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
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category,
 * cudf::nvstring_category
 * Cuda natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T>
__forceinline__ __device__
T atomicMax(T* address, T val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceMax{});
}

/* Overloads for `atomicCAS` */
/** --------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory,
 * computes (`old` == `compare` ? `val` : `old`),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomicCAS` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category, cudf::nvstring_category
 * cudf::bool8
 * Cuda natively supports `sint32`, `uint32`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T>
__forceinline__ __device__
T atomicCAS(T* address, T compare, T val)
{
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}


/* Overloads for `atomicAnd` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory,
 * computes (old & val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported types for `atomicAnd` are:
 *   singed/unsigned integer 8/16/32/64 bits
 * Cuda natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__
T atomicAnd(T* address, T val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceAnd{});
}

/* Overloads for `atomicOr` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory,
 * computes (old | val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported types for `atomicOr` are:
 *   singed/unsigned integer 8/16/32/64 bits
 * Cuda natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__
T atomicOr(T* address, T val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceOr{});
}

/* Overloads for `atomicXor` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory,
 * computes (old ^ val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported types for `atomicXor` are:
 *   singed/unsigned integer 8/16/32/64 bits
 * Cuda natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
__forceinline__ __device__
T atomicXor(T* address, T val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceXor{});
}

#endif
