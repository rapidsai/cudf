#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/wrapper_types.hpp"
#include "utilities/error_utils.hpp"

#include "utilities/device_atomics.cuh"

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
    typedef IdentityLoader Loader;
    typedef cudf::DeviceSum Op;
};

struct ReductionProduct {
    typedef IdentityLoader Loader;
    typedef cudf::DeviceProduct Op;
};

struct ReductionSumOfSquares {
    typedef SquaredLoader Loader;
    typedef cudf::DeviceSum Op;
};

struct ReductionMin{
    typedef IdentityLoader Loader;
    typedef cudf::DeviceMin Op;
};

struct ReductionMax{
    typedef IdentityLoader Loader;
    typedef cudf::DeviceMax Op;
};

} // namespace reductions
} // namespace cudf

