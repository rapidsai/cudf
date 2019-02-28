#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"

namespace cudf {
namespace reduction {


	template <typename T, typename T_equivalent, typename Op>
    __forceinline__  __device__
    void update_existing_value(T_equivalent & existing_value, value_type const & insert_pair, Op op)
    {
      const mapped_type insert_value = insert_pair.second;

      mapped_type old_value = existing_value;

      mapped_type expected{old_value};

      // Attempt to perform the aggregation with existing_value and
      // store the result atomically
      do 
      {
        expected = old_value;

        const mapped_type new_value = op(insert_value, old_value);

        old_value = atomicCAS(&existing_value, expected, new_value);
      }
      // Guard against another thread's update to existing_value
      while( expected != old_value );
    }

    // TODO Overload atomicAdd for 1 byte and 2 byte types, until then, overload specifically for the types
    // where atomicAdd already has an overload. Otherwise the generic update_existing_value will be used.
    // Specialization for COUNT aggregator
    __forceinline__ __host__ __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, count_op<int32_t> op)
    {
      atomicAdd(&existing_value, static_cast<mapped_type>(1));
    }
    // Specialization for COUNT aggregator
    __forceinline__ __host__ __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, count_op<int64_t> op)
    {
      atomicAdd(&existing_value, static_cast<mapped_type>(1));
    }
    // Specialization for COUNT aggregator
    __forceinline__ __host__ __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, count_op<float> op)
    {
      atomicAdd(&existing_value, static_cast<mapped_type>(1));
    }
    // Specialization for COUNT aggregator
    __forceinline__ __host__ __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, count_op<double> op)
    {
      atomicAdd(&existing_value, static_cast<mapped_type>(1));
    }

    // Specialization for SUM aggregator (int32)
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, sum_op<int32_t> op)
    {
      atomicAdd(&existing_value, insert_pair.second);
    }

    // Specialization for SUM aggregator (int64)
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, sum_op<int64_t> op)
    {
      atomicAdd(&existing_value, insert_pair.second);
    }

    // Specialization for SUM aggregator (fp32)
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, sum_op<float> op)
    {
      atomicAdd(&existing_value, insert_pair.second);
    }

    // Specialization for SUM aggregator (fp64)
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, sum_op<double> op)
    {
      atomicAdd(&existing_value, insert_pair.second);
    }

















struct IdentityLoader {
    template<typename T>
    __device__
        T operator() (const T *ptr, int pos) const {
        return ptr[pos];
    }
};

struct DeviceSum {
    typedef IdentityLoader Loader;
    typedef DeviceSum second;

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
    typedef DeviceProduct second;

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
    // round 2 just uses the basic sum reduction
    typedef DeviceSum second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) const {
        return lhs + rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

struct DeviceForNonArithmetic {};

struct DeviceMin : DeviceForNonArithmetic {
    typedef IdentityLoader Loader;
    typedef DeviceMin second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs <= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
};

struct DeviceMax : DeviceForNonArithmetic {
    typedef IdentityLoader Loader;
    typedef DeviceMax second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs >= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::lowest(); }
};


} // namespace reduction
} // namespace cudf



