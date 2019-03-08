#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"

namespace cudf {
namespace reduction {

// force reinterpret cast
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
// specializations

template <>
__forceinline__  __device__
float genericAtomicCAS(
    float* addr, float const & expected, float const & new_value)
{
    return genericAtomicCAS_type<float, unsigned int>(
        addr, expected, new_value);
}

template <>
__forceinline__  __device__
double genericAtomicCAS(
    double* addr, double const & expected, double const & new_value)
{
    return genericAtomicCAS_type<double, unsigned long long int>(
        addr, expected, new_value);
}


// int8_t/int16_t assumes that the address of addr must aligned with int32_t
// need align free genericAtomicCAS for int8_t/int16_t
template <>
__forceinline__  __device__
int8_t genericAtomicCAS(
    int8_t* addr, int8_t const & expected, int8_t const & new_value)
{
    return genericAtomicCAS_type<int8_t, unsigned int>(
        addr, expected, new_value);
}

template <>
__forceinline__  __device__
int16_t genericAtomicCAS(
    int16_t* addr, int16_t const & expected, int16_t const & new_value)
{
    return genericAtomicCAS_type<int16_t, unsigned int>(
        addr, expected, new_value);
}

template <>
__forceinline__  __device__
int64_t genericAtomicCAS(
    int64_t* addr, int64_t const & expected, int64_t const & new_value)
{
    return genericAtomicCAS_type<int64_t, unsigned long long int>(
        addr, expected, new_value);
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

template <>
__forceinline__  __device__
void genericAtomicOperation(
    int32_t& existing_value, int32_t const & update_value, DeviceSum op)
{
    atomicAdd(&existing_value, update_value);
}

template <>
__forceinline__  __device__
void genericAtomicOperation(
    float& existing_value, float const & update_value, DeviceSum op)
{
    atomicAdd(&existing_value, update_value);
}

template <>
__forceinline__  __device__
void genericAtomicOperation(
    double& existing_value, double const & update_value, DeviceSum op)
{
    atomicAdd(&existing_value, update_value);
}

template <>
__forceinline__  __device__
void genericAtomicOperation(
    int32_t& existing_value, int32_t const & update_value, DeviceMin op)
{
    atomicMin(&existing_value, update_value);
}

template <>
__forceinline__  __device__
void genericAtomicOperation(
    int64_t& existing_value, int64_t const & update_value, DeviceMin op)
{
    atomicMin(reinterpret_cast<long long*>(&existing_value),
        static_cast<long long>(update_value));
}

template <>
__forceinline__  __device__
void genericAtomicOperation(
    int32_t& existing_value, int32_t const & update_value, DeviceMax op)
{
    atomicMax(&existing_value, update_value);
}

template <>
__forceinline__  __device__
void genericAtomicOperation(
    int64_t& existing_value, int64_t const & update_value, DeviceMax op)
{
    atomicMax(reinterpret_cast<long long*>(&existing_value),
        static_cast<long long>(update_value));
}

} // namespace reduction
} // namespace cudf



