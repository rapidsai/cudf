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

  template <datetime_component DatetimeComponent>
  struct extract_component_operator {

    column_device_view column;

    CUDA_HOST_DEVICE_CALLABLE extract_component_operator(column_device_view col) : column(col) {}

    template <typename Element>
    typename std::enable_if_t<not cudf::is_timestamp_t<Element>::value, int16_t>
    CUDA_HOST_DEVICE_CALLABLE operator()(size_type const i) const { return 0; }

    template <typename Element>
    typename std::enable_if_t<cudf::is_timestamp_t<Element>::value, int16_t>
    CUDA_HOST_DEVICE_CALLABLE operator()(size_type const i) const {

      using namespace simt::std::chrono;
      using Duration = typename Element::duration;

      auto ts = column.element<Element>(i);
      auto days_since_epoch = floor<days>(ts);

      switch (DatetimeComponent) {
        case cudf::datetime::detail::year: return static_cast<int>(year_month_day(days_since_epoch).year());
        case cudf::datetime::detail::month: return static_cast<unsigned>(year_month_day(days_since_epoch).month());
        case cudf::datetime::detail::day: return static_cast<unsigned>(year_month_day(days_since_epoch).day());
        case cudf::datetime::detail::weekday: return year_month_weekday(days_since_epoch).weekday().iso_encoding();
        default: break;
      }

      auto time_since_midnight = ts - days_since_epoch;

      if (time_since_midnight.count() < 0) {
        time_since_midnight += days(1);
      }

      auto hrs_ = duration_cast<hours>(time_since_midnight);
      auto mins_ = duration_cast<minutes>(time_since_midnight - hrs_);
      auto secs_ = duration_cast<seconds>(time_since_midnight - hrs_ - mins_);

      switch (DatetimeComponent) {
        case cudf::datetime::detail::hour: return hrs_.count();
        case cudf::datetime::detail::minute: return mins_.count();
        case cudf::datetime::detail::second: return secs_.count();
        default: return 0;
      }
    }
  };

  template <datetime_component DatetimeComponent>
  struct launch_extract_component {

    column_device_view input;
    mutable_column_view output;

    launch_extract_component(column_device_view inp, mutable_column_view out) : input(inp), output(out) {}

    template <typename Element>
    typename std::enable_if_t<not cudf::is_timestamp_t<Element>::value, void>
    operator()(cudaStream_t stream) {
      CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
    }

    template <typename Element>
    typename std::enable_if_t<cudf::is_timestamp_t<Element>::value, void>
    operator()(cudaStream_t stream) {
      auto functor = extract_component_operator<DatetimeComponent>{input};
      auto bound_f = [=] __device__ (size_type const i) {
        return functor.template operator()<Element>(i);
      };
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                       output.begin<int16_t>(),
                       output.end<int16_t>(),
                       bound_f);
    }
  };

  template <datetime_component DatetimeComponent>
  std::unique_ptr<column> extract_component(column_view const& input, cudaStream_t stream) {

    auto null_mask_state = input.nullable() ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;
    auto output = make_numeric_column(data_type{type_id::INT16},
                                      input.size(), null_mask_state,
                                      stream, rmm::mr::get_default_resource());

    auto d_input = column_device_view::create(input);
    auto m_output = static_cast<mutable_column_view>(*output);
    auto launch = launch_extract_component<DatetimeComponent>{*d_input, m_output};

    if (null_mask_state == mask_state::UNINITIALIZED) {
      CUDA_TRY(cudaMemcpy(m_output.null_mask(), input.null_mask(),
                          input.size() * sizeof(bitmask_type),
                          cudaMemcpyDefault));
    }

    experimental::type_dispatcher(input.type(), launch, stream);

    return std::move(output);
  }
} // detail

std::unique_ptr<column> extract_year(column_view const& column, cudaStream_t stream) {
  return std::move(detail::extract_component<detail::year>(column, stream));
}

std::unique_ptr<column> extract_month(column_view const& column, cudaStream_t stream) {
  return std::move(detail::extract_component<detail::month>(column, stream));
}

std::unique_ptr<column> extract_day(column_view const& column, cudaStream_t stream) {
  return std::move(detail::extract_component<detail::day>(column, stream));
}

std::unique_ptr<column> extract_weekday(column_view const& column, cudaStream_t stream) {
  return std::move(detail::extract_component<detail::weekday>(column, stream));
}

std::unique_ptr<column> extract_hour(column_view const& column, cudaStream_t stream) {
  return std::move(detail::extract_component<detail::hour>(column, stream));
}

std::unique_ptr<column> extract_minute(column_view const& column, cudaStream_t stream) {
  return std::move(detail::extract_component<detail::minute>(column, stream));
}

std::unique_ptr<column> extract_second(column_view const& column, cudaStream_t stream) {
  return std::move(detail::extract_component<detail::second>(column, stream));
}

} // datetime
} // cudf
