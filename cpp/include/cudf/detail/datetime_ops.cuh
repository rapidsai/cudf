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
    if (!ymd.ok()) ymd = ymd.year() / ymd.month() / last;

    // Put back the time component to the date
    return sys_days{ymd} + (time_val - days_since_epoch);
  }
};
}  // namespace detail
}  // namespace datetime
}  // namespace cudf
