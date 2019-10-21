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
  struct extract_component_op {

    template <typename element_t>
    typename std::enable_if_t<not cudf::is_timestamp_t<element_t>::value, int16_t>
    __device__ operator()(element_t const ts) const { return 0; }

    template <typename timestamp_t>
    typename std::enable_if_t<cudf::is_timestamp_t<timestamp_t>::value, int16_t>
    __device__ operator()(timestamp_t const ts) const {

      using namespace simt::std::chrono;

      auto time_since_epoch = ts.time_since_epoch();
      auto days_since_epoch = sys_days(time_since_epoch);

      if (component == cudf::datetime::detail::year) { return static_cast<int>(year_month_day{days_since_epoch}.year()); }
      if (component == cudf::datetime::detail::month) { return static_cast<unsigned>(year_month_day{days_since_epoch}.month()); }
      if (component == cudf::datetime::detail::day) { return static_cast<unsigned>(year_month_day{days_since_epoch}.day()); }
      if (component == cudf::datetime::detail::weekday) { return year_month_weekday{days_since_epoch}.weekday().iso_encoding(); }

      auto days_since_epoch_clamped = time_since_epoch.count() < 0
        ? time_point_cast<timestamp_t>(ceil<days>(days_since_epoch))
        : time_point_cast<timestamp_t>(floor<days>(days_since_epoch));

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
  struct extract_component {

    cudf::column_view input;
    cudf::mutable_column_view output;
    extract_component(column_view const& _input,
                      mutable_column_view _output)
                      : input(_input), output(_output) {}

    template <typename element_t>
    typename std::enable_if_t<not cudf::is_timestamp_t<element_t>::value, void>
    __host__ __device__ operator()() {}

    template <typename timestamp_t>
    typename std::enable_if_t<cudf::is_timestamp_t<timestamp_t>::value, void>
    __host__ __device__ operator()() {

      cudaStream_t stream = 0;

      thrust::transform(
        rmm::exec_policy(stream)->on(stream),
        input.begin<timestamp_t>(), input.end<timestamp_t>(),
        output.begin<int16_t>(), extract_component_op<component>{}
      );
    }
  };
} // detail

std::unique_ptr<cudf::column> extract_year(cudf::column_view const& input) {
  cudaStream_t stream = 0;
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                          input.size(), mask_state::UNINITIALIZED,
                                          stream, rmm::mr::get_default_resource());
  cudf::experimental::type_dispatcher(input.type(),
                                      detail::extract_component<cudf::datetime::detail::month>{
                                        input, *output
                                      });
  return std::move(output);
}

std::unique_ptr<cudf::column> extract_month(cudf::column_view const& input) {
  cudaStream_t stream = 0;
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                          input.size(), mask_state::UNINITIALIZED,
                                          stream, rmm::mr::get_default_resource());
  cudf::experimental::type_dispatcher(input.type(),
                                      detail::extract_component<cudf::datetime::detail::month>{
                                        input, *output
                                      });
  return std::move(output);
}

std::unique_ptr<cudf::column> extract_day(cudf::column_view const& input) {
  cudaStream_t stream = 0;
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                          input.size(), mask_state::UNINITIALIZED,
                                          stream, rmm::mr::get_default_resource());
  cudf::experimental::type_dispatcher(input.type(),
                                      detail::extract_component<cudf::datetime::detail::day>{
                                        input, *output
                                      });
  return std::move(output);
}

std::unique_ptr<cudf::column> extract_weekday(cudf::column_view const& input) {
  cudaStream_t stream = 0;
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                          input.size(), mask_state::UNINITIALIZED,
                                          stream, rmm::mr::get_default_resource());
  cudf::experimental::type_dispatcher(input.type(),
                                      detail::extract_component<cudf::datetime::detail::weekday>{
                                        input, *output
                                      });
  return std::move(output);
}

std::unique_ptr<cudf::column> extract_hour(cudf::column_view const& input) {
  cudaStream_t stream = 0;
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                          input.size(), mask_state::UNINITIALIZED,
                                          stream, rmm::mr::get_default_resource());
  cudf::experimental::type_dispatcher(input.type(),
                                      detail::extract_component<cudf::datetime::detail::hour>{
                                        input, *output
                                      });
  return std::move(output);
}

std::unique_ptr<cudf::column> extract_minute(cudf::column_view const& input) {
  cudaStream_t stream = 0;
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                          input.size(), mask_state::UNINITIALIZED,
                                          stream, rmm::mr::get_default_resource());
  cudf::experimental::type_dispatcher(input.type(),
                                      detail::extract_component<cudf::datetime::detail::minute>{
                                        input, *output
                                      });
  return std::move(output);
}

std::unique_ptr<cudf::column> extract_second(cudf::column_view const& input) {
  cudaStream_t stream = 0;
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                          input.size(), mask_state::UNINITIALIZED,
                                          stream, rmm::mr::get_default_resource());
  cudf::experimental::type_dispatcher(input.type(),
                                      detail::extract_component<cudf::datetime::detail::second>{
                                        input, *output
                                      });
  return std::move(output);
}

} // datetime
} // cudf
