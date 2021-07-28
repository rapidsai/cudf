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
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
namespace cudf {
namespace datetime {
namespace detail {

// Number of days until month indexed by leap year and month (0-based index)
static __device__ int16_t const days_until_month[2][13] = {
  {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},  // For non leap years
  {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}   // For leap years
};

CUDA_DEVICE_CALLABLE uint8_t days_in_month(cuda::std::chrono::month mon, bool is_leap_year)
{
  return days_until_month[is_leap_year][unsigned{mon}] -
         days_until_month[is_leap_year][unsigned{mon} - 1];
}

// std chrono implementation is copied here due to nvcc bug 2909685
// https://howardhinnant.github.io/date_algorithms.html#days_from_civil
CUDA_DEVICE_CALLABLE timestamp_D compute_sys_days(cuda::std::chrono::year_month_day const& ymd)
{
  const int yr       = static_cast<int>(ymd.year()) - (ymd.month() <= cuda::std::chrono::month{2});
  const unsigned mth = static_cast<unsigned>(ymd.month());
  const unsigned dy  = static_cast<unsigned>(ymd.day());

  const int era      = (yr >= 0 ? yr : yr - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(yr - era * 400);                // [0, 399]
  const unsigned doy = (153 * (mth + (mth > 2 ? -3 : 9)) + 2) / 5 + dy - 1;  // [0, 365]
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;                // [0, 146096]
  return timestamp_D{duration_D{era * 146097 + static_cast<int>(doe) - 719468}};
}

struct add_calendrical_months_functor_impl {
  template <typename Timestamp, typename MonthType>
  Timestamp __device__ operator()(Timestamp time_val, MonthType months_val)
  {
    using namespace cuda::std::chrono;
    using duration_m = duration<int32_t, months::period>;

    // Get the days component from the input
    auto days_since_epoch = floor<days>(time_val);

    // Add the number of months
    year_month_day ymd{days_since_epoch};
    ymd += duration_m{months_val};

    // If the new date isn't valid, scale it back to the last day of the
    // month.
    // IDEAL: if (!ymd.ok()) ymd = ymd.year()/ymd.month()/last;
    auto month_days = days_in_month(ymd.month(), ymd.year().is_leap());
    if (unsigned{ymd.day()} > month_days) ymd = ymd.year() / ymd.month() / day{month_days};

    // Put back the time component to the date
    return
      // IDEAL: sys_days{ymd} + ...
      compute_sys_days(ymd) + (time_val - days_since_epoch);
  }
};
}  // namespace detail
}  // namespace datetime
}  // namespace cudf
