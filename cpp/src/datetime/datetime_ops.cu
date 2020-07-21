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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/thrust_rmm_allocator.h>

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
};

template <datetime_component Component>
struct extract_component_operator {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE int16_t operator()(Timestamp const ts) const
  {
    using namespace simt::std::chrono;

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

CUDA_DEVICE_CALLABLE auto days_in_month(simt::std::chrono::month mon, bool is_leap_year) -> uint8_t
{
  using namespace simt::std::chrono;
  // The expression in switch has to be integral/enumerated type.
  // The constexpr in case has to match the switch type
  switch (unsigned{mon}) {
    case unsigned{January}: return 31;
    case unsigned{February}: return is_leap_year ? 29 : 28;
    case unsigned{March}: return 31;
    case unsigned{April}: return 30;
    case unsigned{May}: return 31;
    case unsigned{June}: return 30;
    case unsigned{July}: return 31;
    case unsigned{August}: return 31;
    case unsigned{September}: return 30;
    case unsigned{October}: return 31;
    case unsigned{November}: return 30;
    case unsigned{December}: return 31;
    default: return 0;
  }
}

// Round up the date to the last day of the month and return the
// date only (without the time component)
struct extract_last_day_of_month {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE timestamp_D operator()(Timestamp const ts) const
  {
    using namespace simt::std::chrono;
    // IDEAL: does not work with CUDA10.0 due to nvcc compiler bug
    // cannot invoke ym_last_day.day()
    // const year_month_day orig_ymd(floor<days>(ts));
    // const year_month_day_last ym_last_day(orig_ymd.year(), month_day_last(orig_ymd.month()));
    // return timestamp_D(sys_days(ym_last_day));

    // Only has the days - time component is chopped off, which is what we want
    auto const days_since_epoch = floor<days>(ts);
    auto const date             = year_month_day(days_since_epoch);
    auto const last_day         = days_in_month(date.month(), date.year().is_leap());

    return timestamp_D(days_since_epoch + days(last_day - static_cast<unsigned>(date.day())));
  }
};

// Number of days until month indexed by leap year and month (0-based index)
static __device__ int16_t const days_until_month[2][12] = {
  {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},  // For non leap years
  {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335}   // For leap years
};

// Extract the day number of the year present in the timestamp
struct extract_day_num_of_year {
  template <typename Timestamp>
  CUDA_DEVICE_CALLABLE int16_t operator()(Timestamp const ts) const
  {
    using namespace simt::std::chrono;

    // Only has the days - time component is chopped off, which is what we want
    auto const days_since_epoch = floor<days>(ts);
    auto const date             = year_month_day(days_since_epoch);

    return days_until_month[date.year().is_leap()][unsigned{date.month()} - 1] +
           unsigned{date.day()};
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
    cudaStream_t stream) const
  {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <typename Timestamp>
  typename std::enable_if_t<cudf::is_timestamp_t<Timestamp>::value, void> operator()(
    cudaStream_t stream) const
  {
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      input.begin<Timestamp>(),
                      input.end<Timestamp>(),
                      output.begin<OutputColT>(),
                      TransformFunctor{});
  }
};

// Create an output column by applying the functor to every element from the input column
template <typename TransformFunctor, cudf::type_id OutputColCudfT>
std::unique_ptr<column> apply_datetime_op(column_view const& column,
                                          cudaStream_t stream,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_timestamp(column.type()), "Column type should be timestamp");
  auto size            = column.size();
  auto output_col_type = data_type{OutputColCudfT};

  // Return an empty column if source column is empty
  if (size == 0) return make_empty_column(output_col_type);

  auto output = make_fixed_width_column(
    output_col_type, size, copy_bitmask(column, stream, mr), column.null_count(), stream, mr);
  auto launch =
    launch_functor<TransformFunctor, typename cudf::id_to_type_impl<OutputColCudfT>::type>{
      column, static_cast<mutable_column_view>(*output)};

  type_dispatcher(column.type(), launch, stream);

  return output;
}

struct add_months_functor {
  column_view timestamp_column;
  column_view months_column;
  mutable_column_view output;

  add_months_functor(column_view tsc, column_view mc, mutable_column_view out)
    : timestamp_column(tsc), months_column(mc), output(out)
  {
  }

  // std chrono implementation is copied here due to nvcc bug
  // https://howardhinnant.github.io/date_algorithms.html#days_from_civil
  static CUDA_DEVICE_CALLABLE timestamp_D
  compute_sys_days(simt::std::chrono::year_month_day const& ymd)
  {
    const int yr = static_cast<int>(ymd.year()) - (ymd.month() <= simt::std::chrono::month{2});
    const unsigned mth = static_cast<unsigned>(ymd.month());
    const unsigned dy  = static_cast<unsigned>(ymd.day());

    const int era      = (yr >= 0 ? yr : yr - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(yr - era * 400);                // [0, 399]
    const unsigned doy = (153 * (mth + (mth > 2 ? -3 : 9)) + 2) / 5 + dy - 1;  // [0, 365]
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;                // [0, 146096]
    return timestamp_D{duration_D{era * 146097 + static_cast<int>(doe) - 719468}};
  }

  template <typename Element>
  typename std::enable_if_t<!cudf::is_timestamp_t<Element>::value, void> operator()(
    cudaStream_t stream) const
  {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <typename Timestamp>
  typename std::enable_if_t<cudf::is_timestamp_t<Timestamp>::value, void> operator()(
    cudaStream_t stream) const
  {
    auto ts_dev_view  = column_device_view::create(timestamp_column, stream);
    auto d_ts         = *ts_dev_view;
    auto mon_dev_view = column_device_view::create(months_column, stream);
    auto d_mon        = *mon_dev_view;
    auto op_dev_view  = mutable_column_device_view::create(output, stream);
    auto d_op         = *op_dev_view;

    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       timestamp_column.size(),
                       [d_ts, d_mon, d_op] __device__(size_type idx) {
                         if (!d_ts.is_null(idx) && !d_ts.is_null(idx)) {
                           using namespace simt::std::chrono;
                           using duration_m = duration<int32_t, months::period>;

                           // Get the days component from the input
                           auto input_ts         = d_ts.element<Timestamp>(idx);
                           auto days_since_epoch = floor<days>(input_ts);

                           // Add the number of months
                           year_month_day ymd{days_since_epoch};
                           ymd += duration_m{d_mon.element<int16_t>(idx)};

                           // If the new date isn't valid, scale it back to the last day of the
                           // month.
                           // IDEAL: if (!ymd.ok()) ymd = ymd.year()/ymd.month()/last;
                           auto month_days = days_in_month(ymd.month(), ymd.year().is_leap());
                           if (unsigned{ymd.day()} > month_days)
                             ymd = ymd.year() / ymd.month() / day{month_days};

                           // Put back the time component to the date
                           d_op.data<Timestamp>()[idx] =
                             // IDEAL: sys_days{ymd} + ...
                             compute_sys_days(ymd) + (input_ts - days_since_epoch);
                         }
                       });
  }
};

std::unique_ptr<column> add_months(column_view const& timestamp_column,
                                   column_view const& months_column,
                                   cudaStream_t stream,
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

  auto output_col_mask = bitmask_and(table_view({timestamp_column, months_column}), mr, stream);
  auto output          = make_fixed_width_column(
    output_col_type, size, std::move(output_col_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);

  auto launch =
    add_months_functor{timestamp_column, months_column, static_cast<mutable_column_view>(*output)};

  type_dispatcher(timestamp_column.type(), launch, stream);

  return output;
}
}  // namespace detail

std::unique_ptr<column> extract_year(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::YEAR>,
    cudf::type_id::INT16>(column, 0, mr);
}

std::unique_ptr<column> extract_month(column_view const& column,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::MONTH>,
    cudf::type_id::INT16>(column, 0, mr);
}

std::unique_ptr<column> extract_day(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::DAY>,
    cudf::type_id::INT16>(column, 0, mr);
}

std::unique_ptr<column> extract_weekday(column_view const& column,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::WEEKDAY>,
    cudf::type_id::INT16>(column, 0, mr);
}

std::unique_ptr<column> extract_hour(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::HOUR>,
    cudf::type_id::INT16>(column, 0, mr);
}

std::unique_ptr<column> extract_minute(column_view const& column,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::MINUTE>,
    cudf::type_id::INT16>(column, 0, mr);
}

std::unique_ptr<column> extract_second(column_view const& column,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_datetime_op<
    detail::extract_component_operator<detail::datetime_component::SECOND>,
    cudf::type_id::INT16>(column, 0, mr);
}

std::unique_ptr<column> last_day_of_month(column_view const& column,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_datetime_op<detail::extract_last_day_of_month,
                                   cudf::type_id::TIMESTAMP_DAYS>(column, 0, mr);
}

std::unique_ptr<column> day_of_year(column_view const& column, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_datetime_op<detail::extract_day_num_of_year, cudf::type_id::INT16>(
    column, 0, mr);
}

std::unique_ptr<cudf::column> add_months(cudf::column_view const& timestamp_column,
                                         cudf::column_view const& months_column,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_months(timestamp_column, months_column, 0, mr);
}
}  // namespace datetime
}  // namespace cudf
