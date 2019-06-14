#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/wrapper_types.hpp>
#include <utilities/error_utils.hpp>

#include <utilities/device_atomics.cuh>

namespace cudf {
namespace reductions {

// ------------------------------------------------------------------------
// difinitions of device struct for binary operation

struct ReductionSum {
    using Op = cudf::DeviceSum;
};

struct ReductionProduct {
    using Op = cudf::DeviceProduct;
};

struct ReductionSumOfSquares {
    using Op = cudf::DeviceSum;
};

struct ReductionMin{
    using Op = cudf::DeviceMin;
};

struct ReductionMax{
    using Op = cudf::DeviceMax;
};

struct ReductionMean{
    using Op = cudf::DeviceSum;

    template<typename T>
    static constexpr T intermediate() { return T{0}; }
};

struct ReductionVar{
    using Op = cudf::DeviceSum;

    template<typename T>
    static constexpr T intermediate() { return T{0}; }
};

struct ReductionStd{
    using Op = cudf::DeviceSum;

    template<typename T>
    static constexpr T intermediate() { return T{0}; }
};


} // namespace reductions
} // namespace cudf

