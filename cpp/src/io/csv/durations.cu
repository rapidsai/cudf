/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/types.hpp>
#include <strings/convert/utilities.cuh>
#include <strings/utilities.cuh>

namespace cudf {
namespace io {
namespace detail {
namespace csv {

namespace {

// duration components timeparts structure
struct alignas(4) duration_component {
  int32_t day;         //-2,147,483,648 to 2,147,483,647
  int32_t nanosecond;  // 000000000 to 999999999
  int8_t hour;         // 00 to 23
  int8_t minute;       // 00 to 59
  int8_t second;       // 00 to 59
  bool is_negative;    // true/false
};

template <typename T>
__device__ void dissect_duration(T duration, duration_component* timeparts)
{
  timeparts->is_negative = (duration < T{0});
  timeparts->day         = simt::std::chrono::floor<duration_D>(duration).count();

  if (simt::std::is_same<T, duration_D>::value) return;

  // adjust for pandas format
  if (timeparts->is_negative) {
    duration =
      simt::std::chrono::duration_cast<T>(duration % duration_D(1) + simt::std::chrono::hours(24));
  }
  duration_s seconds = simt::std::chrono::duration_cast<duration_s>(duration);
  timeparts->hour =
    (simt::std::chrono::duration_cast<simt::std::chrono::hours>(seconds) % duration_D(1)).count();
  timeparts->minute = (simt::std::chrono::duration_cast<simt::std::chrono::minutes>(seconds) %
                       simt::std::chrono::hours(1))
                        .count();
  timeparts->second = (seconds % simt::std::chrono::minutes(1)).count();
  if (not simt::std::is_same<T, duration_s>::value) {
    timeparts->nanosecond =
      (simt::std::chrono::duration_cast<duration_ns>(duration) % duration_s(1)).count();
  }
}

template <typename T>
struct duration_to_string_size_fn {
  const column_device_view d_durations;

  __device__ size_type operator()(size_type idx)
  {
    if (d_durations.is_null(idx)) return 0;
    auto duration                = d_durations.element<T>(idx);
    duration_component timeparts = {0};  // days, hours, minutes, seconds, nanoseconds(9)
    dissect_duration(duration, &timeparts);
    // [-] %d days [+]HH:MM:SS.mmmuuunnn
    return cudf::strings::detail::count_digits(timeparts.day) + 6 + timeparts.is_negative + 18;
  }
};

template <typename T>
struct duration_to_string_fn : public duration_to_string_size_fn<T> {
  const int32_t* d_offsets;
  char* d_chars;
  using duration_to_string_size_fn<T>::d_durations;

  duration_to_string_fn(const column_device_view d_durations,
                        const int32_t* d_offsets,
                        char* d_chars)
    : duration_to_string_size_fn<T>{d_durations}, d_offsets(d_offsets), d_chars(d_chars)
  {
  }

  __device__ char* int_to_2digitstr(int8_t value, char* str)
  {
    assert(value >= -99 && value <= 99);
    value  = std::abs(value);
    str[0] = '0' + value / 10;
    str[1] = '0' + value % 10;
    return str + 2;
  }

  inline __device__ char* day(char* ptr, duration_component const* timeparts)
  {
    cudf::strings::detail::integer_to_string(timeparts->day, ptr);
    return (ptr + cudf::strings::detail::count_digits(timeparts->day));
  }

  inline __device__ char* hour_24(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(timeparts->hour, ptr);
  }

  inline __device__ char* minute(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(timeparts->minute, ptr);
  }
  inline __device__ char* second(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(timeparts->second, ptr);
  }

  inline __device__ char* nanosecond(char* ptr, duration_component const* timeparts)
  {
    auto value = timeparts->nanosecond;
    *ptr       = '.';
    for (int idx = 9; idx > 0; idx--) {
      *(ptr + idx) = '0' + std::abs(value % 10);
      value /= 10;
    }
    return ptr + 10;
  }

  inline __device__ char* pandas_format(duration_component const* timeparts, char* ptr)
  {
    // if (timeparts->is_negative) *ptr++ = '-';
    ptr = day(ptr, timeparts);
    ptr = cudf::strings::detail::copy_and_increment(ptr, " days ", 6);
    if (timeparts->is_negative) *ptr++ = '+';
    ptr    = hour_24(ptr, timeparts);
    *ptr++ = ':';
    ptr    = minute(ptr, timeparts);
    *ptr++ = ':';
    ptr    = second(ptr, timeparts);
    return nanosecond(ptr, timeparts);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_durations.is_null(idx)) return;
    auto duration                = d_durations.template element<T>(idx);
    duration_component timeparts = {0};  // days, hours, minutes, seconds, nanoseconds(9)
    dissect_duration(duration, &timeparts);
    // convert to characters
    pandas_format(&timeparts, d_chars + d_offsets[idx]);
  }
};

/**
 * @brief This dispatch method is for converting durations into strings.
 *
 * The template function declaration ensures only duration types are used.
 */
struct dispatch_from_durations_fn {
  template <typename T, std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& durations,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    size_type strings_count = durations.size();
    auto column             = column_device_view::create(durations, stream);
    auto d_column           = *column;

    // copy null mask
    rmm::device_buffer null_mask = copy_bitmask(durations, stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int32_t>(0), duration_to_string_size_fn<T>{d_column});
    auto offsets_column = strings::detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
    auto offsets_view  = offsets_column->view();
    auto d_new_offsets = offsets_view.template data<int32_t>();

    // build chars column
    auto const chars_bytes =
      cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
    auto chars_column = strings::detail::create_chars_child_column(
      strings_count, durations.null_count(), chars_bytes, mr, stream);
    auto chars_view = chars_column->mutable_view();
    auto d_chars    = chars_view.template data<char>();

    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       duration_to_string_fn<T>{d_column, d_new_offsets, d_chars});

    //
    return make_strings_column(strings_count,
                               std::move(offsets_column),
                               std::move(chars_column),
                               durations.null_count(),
                               std::move(null_mask),
                               stream,
                               mr);
  }

  // non-duration types throw an exception
  template <typename T, std::enable_if_t<not cudf::is_duration<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::mr::device_memory_resource*,
                                     cudaStream_t) const
  {
    CUDF_FAIL("Values for from_durations function must be a duration type.");
  }
};

}  // namespace

std::unique_ptr<column> pandas_format_durations(column_view const& durations,
                                                cudaStream_t stream,
                                                rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = durations.size();
  if (strings_count == 0) return make_empty_column(data_type{type_id::STRING});

  return type_dispatcher(durations.type(), dispatch_from_durations_fn{}, durations, mr, stream);
}

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
