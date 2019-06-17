#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/wrapper_types.hpp>
#include <utilities/error_utils.hpp>

#include <utilities/device_atomics.cuh>

#include <cmath>

namespace cudf {
namespace reductions {

template<typename T>
struct meanvar_no_count
{
    T value;                /// the value
    T value_squared;        /// the value of squared

    CUDA_HOST_DEVICE_CALLABLE
    meanvar_no_count(T _value=0, T _value_squared=0)
    : value(_value), value_squared(_value_squared)
    {};

    using this_t = cudf::reductions::meanvar_no_count<T>;

    CUDA_HOST_DEVICE_CALLABLE
    this_t operator+(this_t const &rhs) const
    {
        return this_t(
            (this->value + rhs.value),
            (this->value_squared + rhs.value_squared)
        );
    };

    CUDA_HOST_DEVICE_CALLABLE
    bool operator==(this_t const &rhs) const
    {
        return (
            (this->value == rhs.value) &&
            (this->value_squared == rhs.value_squared)
        );
    };
};

template<typename T_element>
struct transformer_meanvar_no_count
{
    using T_output = cudf::reductions::meanvar_no_count<T_element>;

    CUDA_HOST_DEVICE_CALLABLE
    T_output operator() (T_element const & value)
    {
        return T_output(value, value*value);
    };
};

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
    struct Intermediate{
        using IType = T;

        static
        auto get_transformer() {
            return cudf::transformer_squared<T>{};
        };

        static
        T ComputeResult(IType& input, gdf_size_type count, gdf_size_type ddof = 1)
        {
            return (input / count);
        };

    };
};

struct ReductionVar{
    using Op = cudf::DeviceSum;

    template<typename T>
    struct Intermediate{
        using IType = transformer_meanvar_no_count<T>;

        static
        auto get_transformer() {
            return cudf::reductions::transformer_meanvar_no_count<T>{};
        };


        static
        T ComputeResult(IType& input, gdf_size_type count, gdf_size_type ddof = 1)
        {
            T mean = input.value / count;
            T asum = input.value_squared;
            gdf_size_type div = count -ddof;

            T var = asum / div - ((mean * mean) * count) /div;
            return var;
        };
    };
};

struct ReductionStd{
    using Op = cudf::DeviceSum;

    template<typename T>
    struct Intermediate{
        using IType = transformer_meanvar_no_count<T>;

        static
        auto get_transformer() {
            return cudf::reductions::transformer_meanvar_no_count<T>{};
        };

        static
        T ComputeResult(IType& input, gdf_size_type count, gdf_size_type ddof = 1)
        {
            using intermediateOp = typename ReductionVar::template Intermediate<T>;
            T var = intermediateOp::ComputeResult(input, count, ddof);

            return T{std::sqrt(var)};
        };
    };
};


} // namespace reductions
} // namespace cudf

