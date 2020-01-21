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
#include <cudf/datetime.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace datetime {
namespace detail {

template <typename Timestamp, datetime_component Component>
struct extract_component_operator {
  static_assert(cudf::is_timestamp<Timestamp>(), "");

  CUDA_DEVICE_CALLABLE int16_t operator()(Timestamp const ts) const {
    using namespace simt::std::chrono;

    auto days_since_epoch = floor<days>(ts);

    auto time_since_midnight = ts - days_since_epoch;

    if (time_since_midnight.count() < 0) {
      time_since_midnight += days(1);
    }

    auto hrs_ = duration_cast<hours>(time_since_midnight);
    auto mins_ = duration_cast<minutes>(time_since_midnight - hrs_);
    auto secs_ = duration_cast<seconds>(time_since_midnight - hrs_ - mins_);

    switch (Component) {
      case datetime_component::YEAR:
        return static_cast<int>(year_month_day(days_since_epoch).year());
      case datetime_component::MONTH:
        return static_cast<unsigned>(year_month_day(days_since_epoch).month());
      case datetime_component::DAY:
        return static_cast<unsigned>(year_month_day(days_since_epoch).day());
      case datetime_component::WEEKDAY:
        return year_month_weekday(days_since_epoch).weekday().iso_encoding();
      case datetime_component::HOUR:
        return hrs_.count();
      case datetime_component::MINUTE:
        return mins_.count();
      case datetime_component::SECOND:
        return secs_.count();
      default:
        return 0;
    }
  }
};

template <datetime_component Component>
struct launch_extract_component {
  column_view input;
  mutable_column_view output;

  launch_extract_component(column_view inp, mutable_column_view out)
      : input(inp), output(out) {}

  template <typename Element>
  typename std::enable_if_t<!cudf::is_timestamp_t<Element>::value, void>
  operator()(cudaStream_t stream) {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <typename Timestamp>
  typename std::enable_if_t<cudf::is_timestamp_t<Timestamp>::value, void>
  operator()(cudaStream_t stream) {
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      input.begin<Timestamp>(), input.end<Timestamp>(),
                      output.begin<int16_t>(),
                      extract_component_operator<Timestamp, Component>{});
  }
};

template <datetime_component Component>
std::unique_ptr<column> extract_component(column_view const& column,
                                          cudaStream_t stream,
                                          rmm::mr::device_memory_resource* mr) {
  auto size = column.size();
  auto type = data_type{type_id::INT16};
  auto null_mask = copy_bitmask(column, stream, mr);
  auto output = std::make_unique<cudf::column>(
      type, size, rmm::device_buffer{size * cudf::size_of(type), stream, mr},
      null_mask, column.null_count(),
      std::vector<std::unique_ptr<cudf::column>>{});

  auto launch = launch_extract_component<Component>{
      column, static_cast<mutable_column_view>(*output)};

  experimental::type_dispatcher(column.type(), launch, stream);

  return output;
}
}  // namespace detail

std::unique_ptr<column> extract_year(column_view const& column,
                                     rmm::mr::device_memory_resource* mr) {
  return detail::extract_component<detail::datetime_component::YEAR>(column, 0,
                                                                     mr);
}

std::unique_ptr<column> extract_month(column_view const& column,
                                      rmm::mr::device_memory_resource* mr) {
  return detail::extract_component<detail::datetime_component::MONTH>(column, 0,
                                                                      mr);
}

std::unique_ptr<column> extract_day(column_view const& column,
                                    rmm::mr::device_memory_resource* mr) {
  return detail::extract_component<detail::datetime_component::DAY>(column, 0,
                                                                    mr);
}

std::unique_ptr<column> extract_weekday(column_view const& column,
                                        rmm::mr::device_memory_resource* mr) {
  return detail::extract_component<detail::datetime_component::WEEKDAY>(column,
                                                                        0, mr);
}

std::unique_ptr<column> extract_hour(column_view const& column,
                                     rmm::mr::device_memory_resource* mr) {
  return detail::extract_component<detail::datetime_component::HOUR>(column, 0,
                                                                     mr);
}

std::unique_ptr<column> extract_minute(column_view const& column,
                                       rmm::mr::device_memory_resource* mr) {
  return detail::extract_component<detail::datetime_component::MINUTE>(column,
                                                                       0, mr);
}

std::unique_ptr<column> extract_second(column_view const& column,
                                       rmm::mr::device_memory_resource* mr) {
  return detail::extract_component<detail::datetime_component::SECOND>(column,
                                                                       0, mr);
}

}  // namespace datetime
}  // namespace cudf
