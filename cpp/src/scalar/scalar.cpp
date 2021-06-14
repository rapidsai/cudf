/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <structs/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <string>

namespace cudf {

scalar::scalar(data_type type,
               bool is_valid,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _type(type), _is_valid(is_valid, stream, mr)
{
}

scalar::scalar(scalar const& other,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _type(other.type()), _is_valid(other._is_valid, stream, mr)
{
}

data_type scalar::type() const noexcept { return _type; }

void scalar::set_valid_async(bool is_valid, rmm::cuda_stream_view stream)
{
  _is_valid.set_value_async(is_valid, stream);
}

bool scalar::is_valid(rmm::cuda_stream_view stream) const { return _is_valid.value(stream); }

bool* scalar::validity_data() { return _is_valid.data(); }

bool const* scalar::validity_data() const { return _is_valid.data(); }

string_scalar::string_scalar(std::string const& string,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::STRING), is_valid, stream, mr),
    _data(string.data(), string.size(), stream, mr)
{
}

string_scalar::string_scalar(string_scalar const& other,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : scalar(other, stream, mr), _data(other._data, stream, mr)
{
}

string_scalar::string_scalar(rmm::device_scalar<value_type>& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : string_scalar(data.value(stream), is_valid, stream, mr)
{
}

string_scalar::string_scalar(value_type const& source,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::STRING), is_valid, stream, mr),
    _data(source.data(), source.size_bytes(), stream, mr)
{
}

string_scalar::string_scalar(rmm::device_buffer&& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::STRING), is_valid, stream, mr), _data(std::move(data))
{
}

string_scalar::value_type string_scalar::value(rmm::cuda_stream_view stream) const
{
  return value_type{data(), size()};
}

size_type string_scalar::size() const { return _data.size(); }

const char* string_scalar::data() const { return static_cast<const char*>(_data.data()); }

string_scalar::operator std::string() const { return this->to_string(rmm::cuda_stream_default); }

std::string string_scalar::to_string(rmm::cuda_stream_view stream) const
{
  std::string result;
  result.resize(_data.size());
  CUDA_TRY(cudaMemcpyAsync(
    &result[0], _data.data(), _data.size(), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  return result;
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rep_type value,
                                          numeric::scale_type scale,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
  : scalar{data_type{type_to_id<T>(), static_cast<int32_t>(scale)}, is_valid, stream, mr},
    _data{value, stream, mr}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rep_type value,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
  : scalar{data_type{type_to_id<T>(), 0}, is_valid, stream, mr}, _data{value, stream, mr}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(T value,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
  : scalar{data_type{type_to_id<T>(), value.scale()}, is_valid, stream, mr},
    _data{value.value(), stream, mr}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rmm::device_scalar<rep_type>&& data,
                                          numeric::scale_type scale,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
  : scalar{data_type{type_to_id<T>(), scale}, is_valid, stream, mr}, _data{std::move(data)}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(fixed_point_scalar<T> const& other,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
  : scalar{other, stream, mr}, _data(other._data, stream, mr)
{
}

template <typename T>
typename fixed_point_scalar<T>::rep_type fixed_point_scalar<T>::value(
  rmm::cuda_stream_view stream) const
{
  return _data.value(stream);
}

template <typename T>
T fixed_point_scalar<T>::fixed_point_value(rmm::cuda_stream_view stream) const
{
  return value_type{
    numeric::scaled_integer<rep_type>{_data.value(stream), numeric::scale_type{type().scale()}}};
}

template <typename T>
typename fixed_point_scalar<T>::rep_type* fixed_point_scalar<T>::data()
{
  return _data.data();
}

template <typename T>
typename fixed_point_scalar<T>::rep_type const* fixed_point_scalar<T>::data() const
{
  return _data.data();
}

/**
 * @brief These define the valid fixed-point scalar types.
 *
 * See `is_fixed_point` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class fixed_point_scalar<numeric::decimal32>;
template class fixed_point_scalar<numeric::decimal64>;

namespace detail {

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(T value,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_to_id<T>()), is_valid, stream, mr), _data(value, stream, mr)
{
}

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(rmm::device_scalar<T>&& data,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_to_id<T>()), is_valid, stream, mr), _data{std::move(data)}
{
}

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(fixed_width_scalar<T> const& other,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
  : scalar{other, stream, mr}, _data(other._data, stream, mr)
{
}

template <typename T>
void fixed_width_scalar<T>::set_value(T value, rmm::cuda_stream_view stream)
{
  _data.set_value_async(value, stream);
  this->set_valid_async(true, stream);
}

template <typename T>
T fixed_width_scalar<T>::value(rmm::cuda_stream_view stream) const
{
  return _data.value(stream);
}

template <typename T>
T* fixed_width_scalar<T>::data()
{
  return _data.data();
}

template <typename T>
T const* fixed_width_scalar<T>::data() const
{
  return _data.data();
}

template <typename T>
fixed_width_scalar<T>::operator value_type() const
{
  return this->value(rmm::cuda_stream_default);
}

/**
 * @brief These define the valid fixed-width scalar types.
 *
 * See `is_fixed_width` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class fixed_width_scalar<bool>;
template class fixed_width_scalar<int8_t>;
template class fixed_width_scalar<int16_t>;
template class fixed_width_scalar<int32_t>;
template class fixed_width_scalar<int64_t>;
template class fixed_width_scalar<uint8_t>;
template class fixed_width_scalar<uint16_t>;
template class fixed_width_scalar<uint32_t>;
template class fixed_width_scalar<uint64_t>;
template class fixed_width_scalar<float>;
template class fixed_width_scalar<double>;
template class fixed_width_scalar<timestamp_D>;
template class fixed_width_scalar<timestamp_s>;
template class fixed_width_scalar<timestamp_ms>;
template class fixed_width_scalar<timestamp_us>;
template class fixed_width_scalar<timestamp_ns>;
template class fixed_width_scalar<duration_D>;
template class fixed_width_scalar<duration_s>;
template class fixed_width_scalar<duration_ms>;
template class fixed_width_scalar<duration_us>;
template class fixed_width_scalar<duration_ns>;

}  // namespace detail

template <typename T>
numeric_scalar<T>::numeric_scalar(T value,
                                  bool is_valid,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
  : detail::fixed_width_scalar<T>(value, is_valid, stream, mr)
{
}

template <typename T>
numeric_scalar<T>::numeric_scalar(rmm::device_scalar<T>&& data,
                                  bool is_valid,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
  : detail::fixed_width_scalar<T>(std::forward<rmm::device_scalar<T>>(data), is_valid, stream, mr)
{
}

template <typename T>
numeric_scalar<T>::numeric_scalar(numeric_scalar<T> const& other,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
  : detail::fixed_width_scalar<T>{other, stream, mr}
{
}

/**
 * @brief These define the valid numeric scalar types.
 *
 * See `is_numeric` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class numeric_scalar<bool>;
template class numeric_scalar<int8_t>;
template class numeric_scalar<int16_t>;
template class numeric_scalar<int32_t>;
template class numeric_scalar<int64_t>;
template class numeric_scalar<uint8_t>;
template class numeric_scalar<uint16_t>;
template class numeric_scalar<uint32_t>;
template class numeric_scalar<uint64_t>;
template class numeric_scalar<float>;
template class numeric_scalar<double>;

template <typename T>
chrono_scalar<T>::chrono_scalar(T value,
                                bool is_valid,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
  : detail::fixed_width_scalar<T>(value, is_valid, stream, mr)
{
}

template <typename T>
chrono_scalar<T>::chrono_scalar(rmm::device_scalar<T>&& data,
                                bool is_valid,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
  : detail::fixed_width_scalar<T>(std::forward<rmm::device_scalar<T>>(data), is_valid, stream, mr)
{
}

template <typename T>
chrono_scalar<T>::chrono_scalar(chrono_scalar<T> const& other,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
  : detail::fixed_width_scalar<T>{other, stream, mr}
{
}

/**
 * @brief These define the valid chrono scalar types.
 *
 * See `is_chrono` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class chrono_scalar<timestamp_D>;
template class chrono_scalar<timestamp_s>;
template class chrono_scalar<timestamp_ms>;
template class chrono_scalar<timestamp_us>;
template class chrono_scalar<timestamp_ns>;
template class chrono_scalar<duration_D>;
template class chrono_scalar<duration_s>;
template class chrono_scalar<duration_ms>;
template class chrono_scalar<duration_us>;
template class chrono_scalar<duration_ns>;

template <typename T>
duration_scalar<T>::duration_scalar(rep_type value,
                                    bool is_valid,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
  : chrono_scalar<T>(T{value}, is_valid, stream, mr)
{
}

template <typename T>
duration_scalar<T>::duration_scalar(duration_scalar<T> const& other,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
  : chrono_scalar<T>{other, stream, mr}
{
}

template <typename T>
typename duration_scalar<T>::rep_type duration_scalar<T>::count()
{
  return this->value().count();
}

/**
 * @brief These define the valid duration scalar types.
 *
 * See `is_duration` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class duration_scalar<duration_D>;
template class duration_scalar<duration_s>;
template class duration_scalar<duration_ms>;
template class duration_scalar<duration_us>;
template class duration_scalar<duration_ns>;

template <typename T>
typename timestamp_scalar<T>::rep_type timestamp_scalar<T>::ticks_since_epoch()
{
  return this->value().time_since_epoch().count();
}

/**
 * @brief These define the valid timestamp scalar types.
 *
 * See `is_timestamp` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class timestamp_scalar<timestamp_D>;
template class timestamp_scalar<timestamp_s>;
template class timestamp_scalar<timestamp_ms>;
template class timestamp_scalar<timestamp_us>;
template class timestamp_scalar<timestamp_ns>;

template <typename T>
template <typename D>
timestamp_scalar<T>::timestamp_scalar(D const& value,
                                      bool is_valid,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
  : chrono_scalar<T>(T{typename T::duration{value}}, is_valid, stream, mr)
{
}

template <typename T>
timestamp_scalar<T>::timestamp_scalar(timestamp_scalar<T> const& other,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
  : chrono_scalar<T>{other, stream, mr}
{
}

#define TS_CTOR(TimestampType, DurationType)                  \
  template timestamp_scalar<TimestampType>::timestamp_scalar( \
    DurationType const&, bool, rmm::cuda_stream_view, rmm::mr::device_memory_resource*);

/**
 * @brief These are the valid combinations of duration types to timestamp types.
 */
TS_CTOR(timestamp_D, duration_D)
TS_CTOR(timestamp_D, int32_t)
TS_CTOR(timestamp_s, duration_D)
TS_CTOR(timestamp_s, duration_s)
TS_CTOR(timestamp_s, int64_t)
TS_CTOR(timestamp_ms, duration_D)
TS_CTOR(timestamp_ms, duration_s)
TS_CTOR(timestamp_ms, duration_ms)
TS_CTOR(timestamp_ms, int64_t)
TS_CTOR(timestamp_us, duration_D)
TS_CTOR(timestamp_us, duration_s)
TS_CTOR(timestamp_us, duration_ms)
TS_CTOR(timestamp_us, duration_us)
TS_CTOR(timestamp_us, int64_t)
TS_CTOR(timestamp_ns, duration_D)
TS_CTOR(timestamp_ns, duration_s)
TS_CTOR(timestamp_ns, duration_ms)
TS_CTOR(timestamp_ns, duration_us)
TS_CTOR(timestamp_ns, duration_ns)
TS_CTOR(timestamp_ns, int64_t)

list_scalar::list_scalar(cudf::column_view const& data,
                         bool is_valid,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::LIST), is_valid, stream, mr), _data(data, stream, mr)
{
}

list_scalar::list_scalar(cudf::column&& data,
                         bool is_valid,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::LIST), is_valid, stream, mr), _data(std::move(data))
{
}

list_scalar::list_scalar(list_scalar const& other,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
  : scalar{other, stream, mr}, _data(other._data, stream, mr)
{
}

column_view list_scalar::view() const { return _data.view(); }

struct_scalar::struct_scalar(table_view const& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::STRUCT), is_valid, stream, mr), _data(data, stream, mr)
{
  init(is_valid, stream, mr);
}

struct_scalar::struct_scalar(host_span<column_view const> data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::STRUCT), is_valid, stream, mr),
    _data(table_view{std::vector<column_view>{data.begin(), data.end()}}, stream, mr)
{
  init(is_valid, stream, mr);
}

struct_scalar::struct_scalar(table&& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::STRUCT), is_valid, stream, mr), _data(std::move(data))
{
  init(is_valid, stream, mr);
}

table_view struct_scalar::view() const { return _data.view(); }

void struct_scalar::init(bool is_valid,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
{
  table_view tv = static_cast<table_view>(_data);
  CUDF_EXPECTS(
    std::all_of(tv.begin(), tv.end(), [](column_view const& col) { return col.size() == 1; }),
    "Struct scalar inputs must have exactly 1 row");

  // validity pushdown
  if (!is_valid) { superimpose_nulls(stream, mr); }
}

void struct_scalar::superimpose_nulls(rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  // push validity mask down
  std::vector<bitmask_type> host_validity({0});
  auto validity = cudf::detail::make_device_uvector_sync(host_validity, stream, mr);
  auto iter     = thrust::make_counting_iterator(0);
  std::for_each(iter, iter + _data.num_columns(), [&](size_type i) {
    cudf::structs::detail::superimpose_parent_nulls(
      validity.data(), 1, _data.get_column(i), stream, mr);
  });
}

}  // namespace cudf
