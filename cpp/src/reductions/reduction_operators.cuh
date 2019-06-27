#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/wrapper_types.hpp>
#include <utilities/error_utils.hpp>

#include <utilities/device_atomics.cuh>

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

struct SquaredLoader {
    template<typename T>
    __device__
    T operator() (const T* ptr, int pos) const {
        T val = ptr[pos];   // load
        return val * val;   // squared
    }
};

struct ReductionSum {
    using Loader = IdentityLoader;
    using Op = cudf::DeviceSum;
};

struct ReductionProduct {
    using Loader = IdentityLoader;
    using Op = cudf::DeviceProduct;
};

struct ReductionSumOfSquares {
    using Loader = SquaredLoader;
    using Op = cudf::DeviceSum;
};

struct ReductionMin{
    using Loader = IdentityLoader;
    using Op = cudf::DeviceMin;
};

struct ReductionMax{
    using Loader = IdentityLoader;
    using Op = cudf::DeviceMax;
};

} // namespace reductions
} // namespace cudf

