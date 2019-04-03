#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/wrapper_types.hpp"
#include "utilities/error_utils.hpp"

namespace cudf {
namespace reductions {

// ------------------------------------------------------------------------
// difinitions of device struct for binary operation

struct IdentityLoader {
    template<typename T>
    __device__
        T operator() (const T *ptr, int pos) const {
        return ptr[pos];
    }
};

struct DeviceSum {
    typedef IdentityLoader Loader;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs + rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

struct DeviceProduct {
    typedef IdentityLoader Loader;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs * rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{1}; }
};

struct DeviceSumOfSquares {
    struct Loader {
        template<typename T>
        __device__
        T operator() (const T* ptr, int pos) const {
            T val = ptr[pos];   // load
            return val * val;   // squared
        }
    };

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) const {
        return lhs + rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{0}; }
};


struct DeviceMin{
    typedef IdentityLoader Loader;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs <= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
};

struct DeviceMax{
    typedef IdentityLoader Loader;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs >= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::lowest(); }
};


// utility fuction for force type reinterpret
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
    void operator()(T* addr, T const & update_value, Op op)
    {
        using T_int = unsigned int;

        T_int * address_uint32 = reinterpret_cast<T_int *>
            (addr - ((size_t)addr & 3));
        unsigned int shift = (((size_t)addr & 3) * 8);

        T_int old = *address_uint32;
        T_int assumed ;

        do {
            assumed = old;
            T target_value = T((old >> shift) & 0xff);
            T new_value = op(update_value, target_value);

            old = (old & ~(0x000000ff << shift)) | (new_value << shift);
            old = atomicCAS(address_uint32, assumed, old);
        } while (assumed != old);
    }
};

// 2 bytes atomic operation
template<typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 2> {
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, Op op)
    {
        using T_int = unsigned int;

        T_int * address_uint32 = reinterpret_cast<T_int *>
            (addr - ((size_t)addr & 2));
        bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? true : false;

        T_int old = *address_uint32;
        T_int assumed ;

        do {
            assumed = old;
            T target_value = (is_32_align) ? T(old >> 16) : T(old & 0xffff);
            T new_value = op(update_value, target_value);

            old = (is_32_align) ? (old & 0xffff) | (new_value << 16)
                                : (old & 0xffff0000) | new_value;
            old = atomicCAS(address_uint32, assumed, old);
        } while (assumed != old);
    }
};

// 4 bytes atomic operation
template<typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 4> {
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, Op op)
    {
        using T_int = unsigned int;

        T old_value = *addr;
        T assumed {old_value};

        do {
            assumed  = old_value;
            const T new_value = op(update_value, old_value);

            T_int ret = atomicCAS(
                reinterpret_cast<T_int*>(addr),
                type_reinterpret<T_int, T>(assumed),
                type_reinterpret<T_int, T>(new_value));

            old_value = type_reinterpret<T, T_int>(ret);

        } while (assumed != old_value);
    }
};

// 8 bytes atomic operation
template<typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 8> {
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, Op op)
    {
        using T_int = unsigned long long int;

        T old_value = *addr;
        T assumed {old_value};

        do {
            assumed  = old_value;
            const T new_value = op(update_value, old_value);

            T_int ret = atomicCAS(
                reinterpret_cast<T_int*>(addr),
                type_reinterpret<T_int, T>(assumed),
                type_reinterpret<T_int, T>(new_value));
            old_value = type_reinterpret<T, T_int>(ret);

        } while (assumed != old_value);
    }
};

// intermediate functoin resolve underlying data type
template <typename T, typename Op>
__forceinline__  __device__
void genericAtomicOperationUnderlyingType(
    T* addr, T const & update_value, Op op)
{
    genericAtomicOperationImpl<T, Op, sizeof(T)>()
        (addr, update_value, op);
}

// The generic atomic operation with binary operator
template <typename T, typename Op>
__forceinline__  __device__
void genericAtomicOperation(T& existing_value, T const & update_value, Op op)
{
    // type cast to underlying data type for wrapper types
    genericAtomicOperationUnderlyingType(
        & cudf::detail::unwrap(existing_value),
        cudf::detail::unwrap(update_value),
        op
    );
}

// ------------------------------------------------------------------------
// specialized functions for operators
// `atomicAdd` supports int32, float, double (signed int64 is not supproted.)
// `atomicMin`, `atomicMax` support int32_t, int64_t

template<>
struct genericAtomicOperationImpl<float, DeviceSum, 4> {
    using T = float;
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, DeviceSum op)
    {
        atomicAdd(addr, update_value);
    }
};


template<>
struct genericAtomicOperationImpl<double, DeviceSum, 8> {
    using T = double;
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, DeviceSum op)
    {
        atomicAdd(addr, update_value);
    }
};

template<>
struct genericAtomicOperationImpl<int32_t, DeviceSum, 4> {
    using T = int32_t;
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, DeviceSum op)
    {
        atomicAdd(addr, update_value);
    }
};

template<>
struct genericAtomicOperationImpl<int32_t, DeviceMin, 4> {
    using T = int32_t;
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, DeviceMin op)
    {
        atomicMin(addr, update_value);
    }
};

template<>
struct genericAtomicOperationImpl<int32_t, DeviceMax, 4> {
    using T = int32_t;
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, DeviceMax op)
    {
        atomicMax(addr, update_value);
    }
};

template<>
struct genericAtomicOperationImpl<int64_t, DeviceMin, 8> {
    using T = int64_t;
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, DeviceMin op)
    {
        using T_out = long long int;
        atomicMin(reinterpret_cast<T_out*>(addr),
            type_reinterpret<T_out, T>(update_value) );
    }
};

template<>
struct genericAtomicOperationImpl<int64_t, DeviceMax, 8> {
    using T = int64_t;
    __forceinline__  __device__
    void operator()(T* addr, T const & update_value, DeviceMax op)
    {
        using T_out = long long int;
        atomicMax(reinterpret_cast<T_out*>(addr),
            type_reinterpret<T_out, T>(update_value) );
    }
};

} // namespace reduction
} // namespace cudf

