/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace datetime {
namespace detail {

  typedef enum {
    invalid = 0,
    year,
    month,
    day,
    weekday,
    hour,
    minute,
    second,
  } datetime_component;

  template <datetime_component component>
  struct extract_each_component {

    cudf::column_device_view input;
    extract_each_component(cudf::column_device_view _input) : input(_input) {}

    template <typename Element>
    typename std::enable_if_t<not cudf::is_timestamp_t<Element>::value, int16_t>
    __device__ operator()(size_type const i) const { return 0; }

    template <typename Timestamp>
    typename std::enable_if_t<cudf::is_timestamp_t<Timestamp>::value, int16_t>
    __device__ operator()(size_type const i) const {

      using namespace simt::std::chrono;

      auto ts = input.element<Timestamp>(i);
      auto time_since_epoch = ts.time_since_epoch();
      auto days_since_epoch = sys_days(time_since_epoch);

      if (component == cudf::datetime::detail::year) { return static_cast<int>(year_month_day(days_since_epoch).year()); }
      if (component == cudf::datetime::detail::month) { return static_cast<unsigned>(year_month_day(days_since_epoch).month()); }
      if (component == cudf::datetime::detail::day) { return static_cast<unsigned>(year_month_day(days_since_epoch).day()); }
      if (component == cudf::datetime::detail::weekday) { return year_month_weekday(days_since_epoch).weekday().iso_encoding(); }

      auto days_since_epoch_clamped = time_since_epoch.count() < 0
        ? time_point_cast<Timestamp>(ceil<days>(days_since_epoch))
        : time_point_cast<Timestamp>(floor<days>(days_since_epoch));

      auto duration_since_midnight = abs(ts - days_since_epoch_clamped);
      auto hrs_ = duration_cast<hours>(duration_since_midnight);

      if (component == cudf::datetime::detail::hour) {
        hrs_ = make24(hrs_, is_pm(hrs_));
        return static_cast<int16_t>(hrs_.count());
      }

      auto mins_ = duration_cast<minutes>(duration_since_midnight - hrs_);
      if (component == cudf::datetime::detail::minute) { return static_cast<int16_t>(mins_.count()); }

      auto secs_ = duration_cast<seconds>(duration_since_midnight - hrs_ - mins_);
      if (component == cudf::datetime::detail::second) { return static_cast<int16_t>(secs_.count()); }

      return 0;
    }
  };

  template <datetime_component component>
  struct extract_component_impl {

    cudaStream_t stream;
    column_view input;
    mutable_column_view output;

    extract_component_impl(cudaStream_t _stream,
                           column_view const& _input,
                           mutable_column_view _output)
                           : stream(_stream) , input(_input), output(_output) {}

    template <typename Element>
    typename std::enable_if_t<not cudf::is_timestamp_t<Element>::value, void>
    __host__ __device__ operator()() {}

    template <typename Timestamp>
    typename std::enable_if_t<cudf::is_timestamp_t<Timestamp>::value, void>
    __host__ __device__ operator()() {
      auto func = extract_each_component<component>{input};
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                       output.begin<int16_t>(), output.end<int16_t>(),
                       [=] __device__ (size_type const i) {
                         return func.template operator()<Timestamp>(i);
                       });
    }
  };

  template <datetime_component component>
  std::unique_ptr<cudf::column> extract_component(cudf::column_view const& input) {

    cudaStream_t stream = 0;
    auto bitmask_state = input.nullable() ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;
    auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16}, input.size(),
                                            bitmask_state, stream, rmm::mr::get_default_resource());

    auto m_output = static_cast<cudf::mutable_column_view>(*output);

    printf("extract_component: {comp=%d, bitmask_state=%d}\n", component, bitmask_state);

    if (bitmask_state == mask_state::UNINITIALIZED) {
      CUDA_TRY(cudaMemcpy(m_output.null_mask(), input.null_mask(),
                          input.size() * sizeof(bitmask_type),
                          cudaMemcpyDefault));
    }

    cudf::experimental::type_dispatcher(input.type(),
                                        detail::extract_component_impl<component>{
                                          stream, input, m_output
                                        });

    return std::move(output);
  }
} // detail

std::unique_ptr<cudf::column> extract_year(cudf::column_view const& input) {
  return std::move(detail::extract_component<detail::year>(input));
}

std::unique_ptr<cudf::column> extract_month(cudf::column_view const& input) {
  return std::move(detail::extract_component<detail::month>(input));
}

std::unique_ptr<cudf::column> extract_day(cudf::column_view const& input) {
  return std::move(detail::extract_component<detail::day>(input));
}

std::unique_ptr<cudf::column> extract_weekday(cudf::column_view const& input) {
  return std::move(detail::extract_component<detail::weekday>(input));
}

std::unique_ptr<cudf::column> extract_hour(cudf::column_view const& input) {
  return std::move(detail::extract_component<detail::hour>(input));
}

std::unique_ptr<cudf::column> extract_minute(cudf::column_view const& input) {
  return std::move(detail::extract_component<detail::minute>(input));
}

std::unique_ptr<cudf::column> extract_second(cudf::column_view const& input) {
  return std::move(detail::extract_component<detail::second>(input));
}

} // datetime
} // cudf
