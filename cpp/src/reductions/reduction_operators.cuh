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

template <typename T_input, typename T_internal>
__forceinline__  __device__
T_input genericAtomicCAS_type(T_input* addr,
    T_input const & expected, T_input const & new_value)
{
    T_internal ret = atomicCAS(
        reinterpret_cast<T_internal*>(addr),
        type_reinterpret<T_internal, T_input>(expected),
        type_reinterpret<T_internal, T_input>(new_value));
    return type_reinterpret<T_input, T_internal>(ret);
}

// ------------------------------------------------------------------------
// generic atomic CAS
template <typename T>
__forceinline__  __device__
T genericAtomicCAS(T* addr, T const & expected, T const & new_value)
{
    return atomicCAS(addr, expected, new_value);
}

// -------------------
// specializations for `genericAtomicCAS`

#define SPECIALIZE_GENERICATOMICAS(T, T_int) \
template <> \
__forceinline__  __device__ \
T genericAtomicCAS( \
    T* addr, T const & expected, T const & new_value){ \
    return genericAtomicCAS_type<T, T_int>(addr, expected, new_value); \
}

SPECIALIZE_GENERICATOMICAS(float,   unsigned int);
SPECIALIZE_GENERICATOMICAS(double,  unsigned long long int);
SPECIALIZE_GENERICATOMICAS(int64_t, unsigned long long int);

// int8_t/int16_t assumes that the address of addr must aligned with int32_t
// need align free genericAtomicCAS for int8_t/int16_t
SPECIALIZE_GENERICATOMICAS(int8_t,  unsigned int);
SPECIALIZE_GENERICATOMICAS(int16_t, unsigned int);

// specializations for wrapper types
SPECIALIZE_GENERICATOMICAS(cudf::timestamp, unsigned long long int);
SPECIALIZE_GENERICATOMICAS(cudf::date64,    unsigned long long int);
SPECIALIZE_GENERICATOMICAS(cudf::category,  unsigned int);
SPECIALIZE_GENERICATOMICAS(cudf::date32,    unsigned int);

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

template <typename T, typename Op>
__forceinline__  __device__
void genericAtomicOperation(T& existing_value, T const & update_value, Op op)
{
  T old_value = existing_value;
  T expected{old_value};

  // Attempt to perform the aggregation with existing_value and
  // store the result atomically
  do
  {
    expected = old_value;

    const T new_value = op(update_value, old_value);

    old_value = genericAtomicCAS(&existing_value, expected, new_value);
  }
  // Guard against another thread's update to existing_value
  while( expected != old_value );
}

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

