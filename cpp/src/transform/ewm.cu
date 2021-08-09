/*
* Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace detail {

rmm::device_vector<double> ewm_denominator(column_view const& input, double beta) {
    rmm::device_vector<double> output(input.size());

    rmm::device_vector<thrust::pair<double, double>> pairs(input.size());
    rmm::device_vector<thrust::pair<double, double>> result_pairs(input.size());
    thrust::fill(pairs.begin(), pairs.end(), thrust::pair<double, double>(beta, 1.0));


    thrust::inclusive_scan(pairs.begin(), pairs.end(), result_pairs.begin(), [=] __host__ __device__ (thrust::pair<double, double> ci, thrust::pair<double, double> cj)->thrust::pair<double, double> {
            double ci0 = thrust::get<0>(ci);
            double ci1 = thrust::get<1>(ci);
            double cj0 = thrust::get<0>(cj);
            double cj1 = thrust::get<1>(cj);
            return thrust::pair<double, double>(ci0*cj0, ci1 * cj0 + cj1);
        }
    );

    thrust::transform(result_pairs.begin(), result_pairs.end(), output.begin(), [=] __host__ __device__ (thrust::pair<double, double> input)->double {
        return thrust::get<1>(input);
    }
    );

    return output;   

}

rmm::device_vector<double> ewm_numerator(column_view const& input, double beta) {
    rmm::device_vector<double> output(input.size());

    rmm::device_vector<thrust::pair<double, double>> pairs(input.size());
    rmm::device_vector<thrust::pair<double, double>> result_pairs(input.size());
    
    thrust::transform(input.begin<double>(), input.end<double>(), pairs.begin(), [=] __host__ __device__ (double input)->thrust::pair<double, double> {
        return thrust::pair<double, double>(beta, input);
    }
    );
    

    thrust::inclusive_scan(pairs.begin(), pairs.end(), result_pairs.begin(), [=] __host__ __device__ (thrust::pair<double, double> ci, thrust::pair<double, double> cj)->thrust::pair<double, double> {
            double ci0 = thrust::get<0>(ci);
            double ci1 = thrust::get<1>(ci);
            double cj0 = thrust::get<0>(cj);
            double cj1 = thrust::get<1>(cj);

            return thrust::pair<double, double>(ci0*cj0, ci1 * cj0 + cj1);

        }
    );

    thrust::transform(result_pairs.begin(), result_pairs.end(), output.begin(), [=] __host__ __device__ (thrust::pair<double, double> input)->double {
        return thrust::get<1>(input);
    }
    );

    return output;    
}

std::unique_ptr<column> ewm(
column_view const& input, 
double alpha, 
rmm::cuda_stream_view stream, 
rmm::mr::device_memory_resource* mr)
{

CUDF_EXPECTS(input.type() == cudf::data_type{cudf::type_id::FLOAT64}, "Column must be float64 type");
auto output = make_fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64}, input.size());
auto output_mutable_view = output->mutable_view();

auto begin = output_mutable_view.begin<double>();
auto end = output_mutable_view.end<double>();

//thrust::fill(rmm::exec_policy(stream), begin, end, (double)1.0);

rmm::device_vector<double> denominator = ewm_denominator(input, (1.0 - alpha));
rmm::device_vector<double> numerator = ewm_numerator(input, (1.0 - alpha));

thrust::transform(rmm::exec_policy(stream), numerator.begin(), numerator.end(), denominator.begin(), output_mutable_view.begin<double>(), thrust::divides<double>());

return output;
}


}  // namespace detail


std::unique_ptr<column> ewm(
        column_view const& input, 
        double alpha, 
        rmm::mr::device_memory_resource* mr)
{
CUDF_FUNC_RANGE();
return detail::ewm(input, alpha, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
