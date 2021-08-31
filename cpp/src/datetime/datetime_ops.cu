/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/detail/datetime.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace datetime {
namespace detail {
enum class datetime_component {
  INVALID = 0,
  YEAR,
  MONTH,
  DAY,
  WEEKDAY,
  HOUR,
  MINUTE,
  SECOND,
  MILLISECOND,
  MICROSECOND,
  NANOSECOND
};

template <datetime_component Component>
struct extract_component_operator {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE int16_t operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;

    auto days_since_epoch = floor<days>(ts);

    auto time_since_midnight = ts - days_since_epoch;

    if (time_since_midnight.count() < 0) { time_since_midnight += days(1); }

    auto hrs_  = duration_cast<hours>(time_since_midnight);
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
      case datetime_component::HOUR: return hrs_.count();
      case datetime_component::MINUTE: return mins_.count();
      case datetime_component::SECOND: return secs_.count();
      default: return 0;
    }
  }
};

template <datetime_component COMPONENT>
struct ceil_timestamp {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE Timestamp operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;
    // want to use this with D, H, T (minute), S, L (millisecond), U
    switch (COMPONENT) {
      case datetime_component::DAY:
        return time_point_cast<typename Timestamp::duration>(ceil<duration_D>(ts));
      case datetime_component::HOUR:
        return time_point_cast<typename Timestamp::duration>(ceil<duration_h>(ts));
      case datetime_component::MINUTE:
        return time_point_cast<typename Timestamp::duration>(ceil<duration_m>(ts));
      case datetime_component::SECOND:
        return time_point_cast<typename Timestamp::duration>(ceil<duration_s>(ts));
      case datetime_component::MILLISECOND:
        return time_point_cast<typename Timestamp::duration>(ceil<duration_ms>(ts));
      case datetime_component::MICROSECOND:
        return time_point_cast<typename Timestamp::duration>(ceil<duration_us>(ts));
      case datetime_component::NANOSECOND:
        return time_point_cast<typename Timestamp::duration>(ceil<duration_ns>(ts));
      default: cudf_assert(false && "Unexpected resolution");
    }

    return {};
  }
};

// Number of days until month indexed by leap year and month (0-based index)
static __device__ int16_t const days_until_month[2][13] = {
  {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},  // For non leap years
  {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}   // For leap years
};

// Round up the date to the last day of the month and return the
// date only (without the time component)
struct extract_last_day_of_month {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE timestamp_D operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;
    const year_month_day ymd(floor<days>(ts));
    auto const ymdl = year_month_day_last{ymd.year() / ymd.month() / last};
    return timestamp_D{sys_days{ymdl}};
  }
};

// Extract the number of days of the month
// A similar operator to `extract_last_day_of_month`, except this returns
// an integer while the other returns a timestamp.
struct days_in_month_op {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE int16_t operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;
    auto const date = year_month_day(floor<days>(ts));
    auto const ymdl = year_month_day_last(date.year() / date.month() / last);
    return static_cast<int16_t>(unsigned{ymdl.day()});
  }
};

// Extract the day number of the year present in the timestamp
struct extract_day_num_of_year {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE int16_t operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;

    // Only has the days - time component is chopped off, which is what we want
    auto const days_since_epoch = floor<days>(ts);
    auto const date             = year_month_day(days_since_epoch);

    return days_until_month[date.year().is_leap()][unsigned{date.month()} - 1] +
           unsigned{date.day()};
  }
};

// Extract the the quarter to which the timestamp belongs to
struct extract_quarter_op {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE int16_t operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;

    // Only has the days - time component is chopped off, which is what we want
    auto const days_since_epoch = floor<days>(ts);
    auto const date             = year_month_day(days_since_epoch);
    auto const month            = unsigned{date.month()};

    // (x + y - 1) / y = ceil(x/y), where x and y are unsigned. x = month, y = 3
    return (month + 2) / 3;
  }
};

// Returns true if the year is a leap year
struct is_leap_year_op {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE bool operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;
    auto const days_since_epoch = floor<days>(ts);
    auto const date             = year_month_day(days_since_epoch);
    return date.year().is_leap();
  }
};

// Specific function for applying ceil/floor date ops
template <typename TransformFunctor>
struct dispatch_ceil {
  template <typename Timestamp>
  std::enable_if_t<cudf::is_timestamp<Timestamp>(), std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& column,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const
  {
    auto size            = column.size();
    auto output_col_type = data_type{cudf::type_to_id<Timestamp>()};

    // Return an empty column if source column is empty
    if (size == 0) return make_empty_column(output_col_type);

    auto output = make_fixed_width_column(output_col_type,
                                          size,
                                          cudf::detail::copy_bitmask(column, stream, mr),
                                          column.null_count(),
                                          stream,
                                          mr);

    thrust::transform(rmm::exec_policy(stream),
                      column.begin<Timestamp>(),
                      column.end<Timestamp>(),
                      output->mutable_view().begin<Timestamp>(),
                      TransformFunctor{});

    return output;
  }

  template <typename Timestamp, typename... Args>
  std::enable_if_t<!cudf::is_timestamp<Timestamp>(), std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUDF_FAIL("Must be cudf::timestamp");
  }
};

// Apply the functor for every element/row in the input column to create the output column
template <typename TransformFunctor, typename OutputColT>
struct launch_functor {
  column_view input;
  mutable_column_view output;

  launch_functor(column_view inp, mutable_column_view out) : input(inp), output(out) {}

  template <typename Element>
  typename std::enable_if_t<!cudf::is_timestamp_t<Element>::value, void> operator()(
    rmm::cuda_stream_view stream) const
  {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <typename Timestamp>
  typename std::enable_if_t<cudf::is_timestamp_t<Timestamp>::value, void> operator()(
    rmm::cuda_stream_view stream) const
  {
    thrust::transform(rmm::exec_policy(stream),
                      input.begin<Timestamp>(),
                      input.end<Timestamp>(),
                      output.begin<OutputColT>(),
                      TransformFunctor{});
  }
};

// Create an output column by applying the functor to every element from the input column
template <typename TransformFunctor, cudf::type_id OutputColCudfT>
std::unique_ptr<column> apply_datetime_op(column_view const& column,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_timestamp(column.type()), "Column type should be timestamp");
  auto size            = column.size();
  auto output_col_type = data_type{OutputColCudfT};

  // Return an empty column if source column is empty
  if (size == 0) return make_empty_column(output_col_type);

  auto output = make_fixed_width_column(output_col_type,
                                        size,
                                        cudf::detail::copy_bitmask(column, stream, mr),
                                        column.null_count(),
                                        stream,
                                        mr);
  auto launch =
    launch_functor<TransformFunctor, typename cudf::id_to_type_impl<OutputColCudfT>::type>{
      column, static_cast<mutable_column_view>(*output)};

  type_dispatcher(column.type(), launch, stream);

  return output;
}

struct add_calendrical_months_functor {
  column_view timestamp_column;
  column_view months_column;
  mutable_column_view output;

  add_calendrical_months_functor(column_view tsc, column_view mc, mutable_column_view out)
    : timestamp_column(tsc), months_column(mc), output(out)
  {
  }

  template <typename Element>
  typename std::enable_if_t<!cudf::is_timestamp_t<Element>::value, void> operator()(
    rmm::cuda_stream_view stream) const
  {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <typename Timestamp>
  typename std::enable_if_t<cudf::is_timestamp_t<Timestamp>::value, void> operator()(
    rmm::cuda_stream_view stream) const
  {
    thrust::transform(rmm::exec_policy(stream),
                      timestamp_column.begin<Timestamp>(),
                      timestamp_column.end<Timestamp>(),
                      months_column.begin<int16_t>(),
                      output.begin<Timestamp>(),
                      [] __device__(auto time_val, auto months_val) {
                        using namespace cuda::std::chrono;
                        using duration_m = duration<int32_t, months::period>;

                        // Get the days component from the input
                        auto days_since_epoch = floor<days>(time_val);

                        // Add the number of months
                        year_month_day ymd{days_since_epoch};
                        ymd += duration_m{months_val};

                        // If the new date isn't valid, scale it back to the last day of the
                        // month.
                        if (!ymd.ok()) ymd = ymd.year() / ymd.month() / last;

                        // Put back the time component to the date
                        return sys_days{ymd} + (time_val - days_since_epoch);
                      });
  }
};

std::unique_ptr<column> add_calendrical_months(column_view const& timestamp_column,
                                               column_view const& months_column,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_timestamp(timestamp_column.type()), "Column type should be timestamp");
  CUDF_EXPECTS(months_column.type() == data_type{type_id::INT16},
               "Months column type should be INT16");
  CUDF_EXPECTS(timestamp_column.size() == months_column.size(),
               "Timestamp and months column should be of the same size");
  auto size            = timestamp_column.size();
  auto output_col_type = timestamp_column.type();

  // Return an empty column if source column is empty
  if (size == 0) return make_empty_column(output_col_type);

  auto output_col_mask =
    cudf::detail::bitmask_and(table_view({timestamp_column, months_column}), stream, mr);
  auto output = make_fixed_width_column(
    output_col_type, size, std::move(output_col_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);

  auto launch = add_calendrical_months_functor{
    timestamp_column, months_column, static_cast<mutable_column_view>(*output)};

  type_dispatcher(timestamp_column.type(), launch, stream);

  return output;
}

template <datetime_component Component>
std::unique_ptr<column> ceil_general(column_view const& column,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(
    column.type(), dispatch_ceil<detail::ceil_timestamp<Component>>{}, column, stream, mr);
}

std::unique_ptr<column> extract_year(column_view const& column,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::YEAR>,
    cudf::type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_month(column_view const& column,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::MONTH>,
    cudf::type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_day(column_view const& column,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::DAY>,
    cudf::type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_weekday(column_view const& column,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::WEEKDAY>,
    cudf::type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_hour(column_view const& column,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::HOUR>,
    cudf::type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_minute(column_view const& column,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::MINUTE>,
    cudf::type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_second(column_view const& column,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::SECOND>,
    cudf::type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> last_day_of_month(column_view const& column,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<detail::extract_last_day_of_month,
                                   cudf::type_id::TIMESTAMP_DAYS>(column, stream, mr);
}

std::unique_ptr<column> day_of_year(column_view const& column,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return detail::apply_datetime_op<detail::extract_day_num_of_year, cudf::type_id::INT16>(
    column, stream, mr);
}

std::unique_ptr<column> is_leap_year(column_view const& column,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return apply_datetime_op<is_leap_year_op, type_id::BOOL8>(column, stream, mr);
}

std::unique_ptr<column> days_in_month(column_view const& column,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  return apply_datetime_op<days_in_month_op, type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_quarter(column_view const& column,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  return apply_datetime_op<extract_quarter_op, type_id::INT16>(column, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> ceil_day(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ceil_general<detail::datetime_component::DAY>(
    column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> ceil_hour(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ceil_general<detail::datetime_component::HOUR>(
    column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> ceil_minute(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ceil_general<detail::datetime_component::MINUTE>(
    column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> ceil_second(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ceil_general<detail::datetime_component::SECOND>(
    column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> ceil_millisecond(column_view const& column,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ceil_general<detail::datetime_component::MILLISECOND>(
    column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> ceil_microsecond(column_view const& column,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ceil_general<detail::datetime_component::MICROSECOND>(
    column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> ceil_nanosecond(column_view const& column,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ceil_general<detail::datetime_component::NANOSECOND>(
    column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_year(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_year(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_month(column_view const& column,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_month(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_day(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_day(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_weekday(column_view const& column,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_weekday(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_hour(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_hour(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_minute(column_view const& column,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_minute(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_second(column_view const& column,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_second(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> last_day_of_month(column_view const& column,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::last_day_of_month(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> day_of_year(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::day_of_year(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamp_column,
                                                     cudf::column_view const& months_column,
                                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_calendrical_months(
    timestamp_column, months_column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> is_leap_year(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_leap_year(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> days_in_month(column_view const& column,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::days_in_month(column, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_quarter(column_view const& column,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_quarter(column, rmm::cuda_stream_default, mr);
}

}  // namespace datetime
}  // namespace cudf
