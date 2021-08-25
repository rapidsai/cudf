#include <cuda/std/chrono>

namespace cudf {
namespace datetime {
namespace detail {

template <typename Timestamp, typename MonthType>
__device__ Timestamp add_calendrical_months_with_scale_back(Timestamp time_val,
                                                            MonthType months_val)
{
  using namespace cuda::std::chrono;

  // Get the days component from the input
  auto days_since_epoch = floor<days>(time_val);

  // Add the number of months
  year_month_day ymd{days_since_epoch};
  ymd += duration<int32_t, months::period>{months_val};

  // If the new date isn't valid, scale it back to the last day of the
  // month.
  if (!ymd.ok()) ymd = ymd.year() / ymd.month() / last;

  // Put back the time component to the date
  return sys_days{ymd} + (time_val - days_since_epoch);
}

}  // namespace detail
}  // namespace datetime
}  // namespace cudf
