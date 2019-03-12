#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/wrapper_types.hpp"
#include "utilities/error_utils.hpp"

namespace cudf {
namespace reduction {

template <typename T_output, typename T_input>
__forceinline__  __device__
T_output type_reinterpret(T_input value)
{
    return *( reinterpret_cast<T_output*>(&value) );
}

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


template <typename T, typename Op>
__forceinline__  __device__
void genericAtomicOperation(T& existing_value, T const & update_value, Op op)
{
    genericAtomicOperationImpl<T, Op, sizeof(T)>() (&existing_value, update_value, op);
}


// ------------------------------------------------------------------------

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


// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// specialized functions for operators
// `atomicAdd` supports int32, float, double (signed int64 is not supproted.)
// `atomicMin`, `atomicMax` support int32_t, int64_t

#define SPECIALIZE_GENERICATOMIOPS(T, Op, AtomicOp) \
template <> \
__forceinline__  __device__ \
void genericAtomicOperation( \
    T& existing_value, T const & update_value, Op op){ \
    AtomicOp(&existing_value, update_value); \
}

#define SPECIALIZE_GENERICATOMIOPS_TYPE(T, Op, AtomicOp, T_int) \
template <> \
__forceinline__  __device__ \
void genericAtomicOperation( \
    T& existing_value, T const & update_value, Op op){ \
    AtomicOp(reinterpret_cast<T_int*>(&existing_value), \
        static_cast<T_int>(update_value)); \
}

SPECIALIZE_GENERICATOMIOPS(int32_t, DeviceSum, atomicAdd);
SPECIALIZE_GENERICATOMIOPS(float  , DeviceSum, atomicAdd);
SPECIALIZE_GENERICATOMIOPS(double,  DeviceSum, atomicAdd);
SPECIALIZE_GENERICATOMIOPS(int32_t, DeviceMin, atomicMin);
SPECIALIZE_GENERICATOMIOPS(int32_t, DeviceMax, atomicMax);

SPECIALIZE_GENERICATOMIOPS_TYPE(int64_t, DeviceMin, atomicMin, long long);
SPECIALIZE_GENERICATOMIOPS_TYPE(int64_t, DeviceMax, atomicMax, long long);

// specializations for wrapper types
SPECIALIZE_GENERICATOMIOPS_TYPE(cudf::category, DeviceSum, atomicAdd, int);
SPECIALIZE_GENERICATOMIOPS_TYPE(cudf::category, DeviceMin, atomicMin, int);
SPECIALIZE_GENERICATOMIOPS_TYPE(cudf::category, DeviceMax, atomicMax, int);
SPECIALIZE_GENERICATOMIOPS_TYPE(cudf::date32, DeviceSum, atomicAdd, int);
SPECIALIZE_GENERICATOMIOPS_TYPE(cudf::date32, DeviceMin, atomicMin, int);
SPECIALIZE_GENERICATOMIOPS_TYPE(cudf::date32, DeviceMax, atomicMax, int);

} // namespace reduction
} // namespace cudf

