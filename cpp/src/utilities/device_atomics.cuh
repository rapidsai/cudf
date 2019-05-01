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
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category.
 * where CUDA atomic operations are, `atomicAdd`, `atomicMin`, `atomicMax`,
 * `atomicCAS`.
 * Also provides `cudf::genericAtomicOperation` which performs atomic operation 
 * with the given binary operator.
 * ---------------------------------------------------------------------------**/

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/wrapper_types.hpp"
#include "utilities/error_utils.hpp"

namespace cudf {

// ------------------------------------------------------------------------
// Binary operators
/* @brief binary `sum` operator */
struct DeviceSum {
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs + rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

/* @brief binary `min` operator */
struct DeviceMin{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs <= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
};

/* @brief binary `max` operator */
struct DeviceMax{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs >= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::lowest(); }
};

/* @brief binary `product` operator */
struct DeviceProduct {
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs * rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{1}; }
};


/* @brief binary `and` operator */
struct DeviceAnd{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return (lhs & rhs );
    }
};

/* @brief binary `or` operator */
struct DeviceOr{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return (lhs | rhs );
    }
};

/* @brief binary `xor` operator */
struct DeviceXor{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return (lhs ^ rhs );
    }
};


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

    // atomicCAS implementation for int8_t
    __forceinline__  __device__
    int8_t atomicCASImpl(int8_t* address, int8_t compare, int8_t val)
    {
        using T = int8_t;
        using T_int = unsigned int;

        T_int shift = ((reinterpret_cast<size_t>(address) & 3) * 8);
        T_int * address_uint32 = reinterpret_cast<T_int *>
            (address - (reinterpret_cast<size_t>(address) & 3));

        // the 'target_value' in `old` can be different from `compare`
        // because other thread may update the value
        // before fetching a value from `address_uint32` in this function
        T_int old = *address_uint32;
        T_int assumed;
        T target_value;
        uint8_t u_val = type_reinterpret<uint8_t, T>(val);

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

    // atomicCAS implementation for int16_t
    __forceinline__  __device__
    int16_t atomicCASImpl(int16_t* address, int16_t compare, int16_t val)
    {
        using T = int16_t;
        using T_int = unsigned int;

        bool is_32_align = (reinterpret_cast<size_t>(address) & 2) ? false : true;
        T_int * address_uint32 = reinterpret_cast<T_int *>
            (reinterpret_cast<size_t>(address) - (is_32_align ? 0 : 2));

        T_int old = *address_uint32;
        T_int assumed;
        T target_value;
        uint16_t u_val = type_reinterpret<uint16_t, T>(val);

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

    // -----------------------------------------------------------------------
    // intermediate function resolve underlying data type
    template <typename T, typename BinaryOp>
    __forceinline__  __device__
    T genericAtomicOperationUnderlyingType(
        T* address, T const & update_value, BinaryOp op)
    {
        T ret = cudf::detail::genericAtomicOperationImpl<T, BinaryOp, sizeof(T)>()
            (address, update_value, op);
        return ret;
    }

} // namespace detail

/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory,
 * computes 'BinaryOp'('old', 'update_value'),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `genericAtomicOperation` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category.
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
    auto ret = cudf::detail::genericAtomicOperationUnderlyingType(
         &cudf::detail::unwrap(*address),
         cudf::detail::unwrap(update_value),
         op);
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
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category.
 * Cuda natively supports `sint32`, `uint32`, `uint64`, `float`, `double,
 * cudf::nvstring_category
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

/**
 * @overload cudf::nvstring_category atomicAdd(cudf::nvstring_category* address, cudf::nvstring_category val)
 */
__forceinline__ __device__
cudf::nvstring_category atomicAdd(cudf::nvstring_category* address, cudf::nvstring_category val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
        (address, val, [](T* a, T v){return atomicAdd(a, v);});
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

/**
 * @overload cudf::nvstring_category atomicMin(cudf::nvstring_category* address, cudf::nvstring_category val)
 */
__forceinline__ __device__
cudf::nvstring_category atomicMin(cudf::nvstring_category* address, cudf::nvstring_category val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
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

/**
 * @overload cudf::nvstring_category atomicMax(cudf::nvstring_category* address, cudf::nvstring_category val)
 */
__forceinline__ __device__
cudf::nvstring_category atomicMax(cudf::nvstring_category* address, cudf::nvstring_category val)
{
    using T = int;
    return cudf::detail::typesAtomicOperation32
        (address, val, [](T* a, T v){return atomicMax(a, v);});
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
 * Cuda natively supports `sint32`, `uint32`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int8_t atomicCAS(int8_t* address, int8_t compare, int8_t val)
{
    return cudf::detail::atomicCASImpl(address, compare, val);
}

/**
 * @overload int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val)
 */
__forceinline__ __device__
int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val)
{
    return cudf::detail::atomicCASImpl(address, compare, val);
}

/**
 * @overload T atomicCAS(T* address, T compare, T val)
 * T belongs to 
 * { 
 * int32_t, int64_t, float, double,
 * cudf::date32, cudf::date64, cudf::timestamp, cudf::category,
 * cudf::nvstring_category
 * }
 */
template <typename T>
__forceinline__ __device__
T atomicCAS(T* address, T compare, T val)
{
    return cudf::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload cudf::bool8 atomicCAS(cudf::bool8* address, cudf::bool8 compare, cudf::bool8 val)
 */
__forceinline__ __device__
cudf::bool8 atomicCAS(cudf::bool8* address, cudf::bool8 compare, cudf::bool8 val)
{
    // TODO: not sure this is the right way...
    return cudf::bool8{atomicCAS(reinterpret_cast<int8_t*>(address), 
                                 cudf::detail::unwrap(compare), 
                                 cudf::detail::unwrap(val))};
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
__forceinline__ __device__
int8_t atomicAnd(int8_t* address, int8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceAnd{});
}

/**
 * @overload uint8_t atomicAnd(uint8_t* address, uint8_t val)
 */
__forceinline__ __device__
uint8_t atomicAnd(uint8_t* address, uint8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceAnd{});
}

/**
 * @overload int16_t atomicAnd(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicAnd(int16_t* address, int16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceAnd{});
}

/**
 * @overload uint16_t atomicAnd(uint16_t* address, uint16_t val)
 */
__forceinline__ __device__
uint16_t atomicAnd(uint16_t* address, uint16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceAnd{});
}

/**
 * @overload int64_t atomicAnd(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicAnd(int64_t* address, int64_t val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicAnd(a, v);});
}

/**
 * @overload uint64_t atomicAnd(uint64_t* address, uint64_t val)
 */
__forceinline__ __device__
uint64_t atomicAnd(uint64_t* address, uint64_t val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicAnd(a, v);});
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
__forceinline__ __device__
int8_t atomicOr(int8_t* address, int8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceOr{});
}

/**
 * @overload uint8_t atomicOr(uint8_t* address, uint8_t val)
 */
__forceinline__ __device__
uint8_t atomicOr(uint8_t* address, uint8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceOr{});
}

/**
 * @overload int16_t atomicOr(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicOr(int16_t* address, int16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceOr{});
}

/**
 * @overload uint16_t atomicOr(uint16_t* address, uint16_t val)
 */
__forceinline__ __device__
uint16_t atomicOr(uint16_t* address, uint16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceOr{});
}

/**
 * @overload int64_t atomicOr(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicOr(int64_t* address, int64_t val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicOr(a, v);});
}

/**
 * @overload uint64_t atomicOr(uint64_t* address, uint64_t val)
 */
__forceinline__ __device__
uint64_t atomicOr(uint64_t* address, uint64_t val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicOr(a, v);});
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
__forceinline__ __device__
int8_t atomicXor(int8_t* address, int8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceXor{});
}

/**
 * @overload uint8_t atomicXor(uint8_t* address, uint8_t val)
 */
__forceinline__ __device__
uint8_t atomicXor(uint8_t* address, uint8_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceXor{});
}

/**
 * @overload int16_t atomicXor(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicXor(int16_t* address, int16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceXor{});
}

/**
 * @overload uint16_t atomicXor(uint16_t* address, uint16_t val)
 */
__forceinline__ __device__
uint16_t atomicXor(uint16_t* address, uint16_t val)
{
    return cudf::genericAtomicOperation(address, val, cudf::DeviceXor{});
}

/**
 * @overload int64_t atomicXor(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicXor(int64_t* address, int64_t val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicXor(a, v);});
}

/**
 * @overload uint64_t atomicXor(uint64_t* address, uint64_t val)
 */
__forceinline__ __device__
uint64_t atomicXor(uint64_t* address, uint64_t val)
{
    using T = long long int;
    return cudf::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicXor(a, v);});
}


#endif
