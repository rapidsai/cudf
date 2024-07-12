/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/prefetch.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>

#include <iostream>

namespace cudf::experimental::prefetch {

namespace detail {

PrefetchConfig& PrefetchConfig::instance()
{
  static PrefetchConfig instance;
  return instance;
}

bool PrefetchConfig::get(std::string_view key)
{
  // Default to not prefetching
  if (config_values.find(key.data()) == config_values.end()) {
    return (config_values[key.data()] = false);
  }
  return config_values[key.data()];
}
void PrefetchConfig::set(std::string_view key, bool value) { config_values[key.data()] = value; }

void prefetch(std::string_view key, void const* ptr, std::size_t size)
{
  if (PrefetchConfig::instance().get(key)) {
    auto result = cudaMemPrefetchAsync(
      ptr, size, rmm::get_current_cuda_device().value(), cudf::get_default_stream().value());
    // Ignore cudaErrorInvalidValue because that will be raised if
    // prefetching is attempted on unmanaged memory.
    if (result != cudaSuccess) {
      if (result == cudaErrorInvalidValue) {
        result = cudaGetLastError();  // clears the error
      } else {
        std::cerr << "Prefetch failed" << std::endl;
        CUDF_CUDA_TRY(result);
      }
    }
    if (PrefetchConfig::instance().debug) {
      std::cerr << "Prefetched " << size << " bytes for key [" << key << "] at location " << ptr
                << "=" << (int)result << std::endl;
    }
  }
}

namespace {
struct prefetch_fn {
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  void operator()(column_view const& col, std::string_view key)
  {
    prefetch(key, col.data<T>(), col.size() * sizeof(T));
  }
  template <typename T, CUDF_ENABLE_IF(not cudf::is_fixed_width<T>())>
  void operator()(column_view const& col, std::string_view key)
  {
    CUDF_UNREACHABLE("");
  }
};
}  // namespace

void prefetch(std::string_view key, cudf::column_view const& col, bool prefetch_mask)
{
  if (col.is_empty()) { return; }
  // only support fixed-width-types
  if (not cudf::is_fixed_width(col.type())) { return; }
  type_dispatcher<dispatch_storage_type>(col.type(), prefetch_fn{}, col, key);
  if (prefetch_mask && col.has_nulls()) {
    prefetch(key, col.null_mask(), cudf::bitmask_allocation_size_bytes(col.size()));
  }
}

}  // namespace detail

void enable_prefetching(std::string_view key) { detail::PrefetchConfig::instance().set(key, true); }

void disable_prefetching(std::string_view key)
{
  detail::PrefetchConfig::instance().set(key, false);
}

void prefetch_debugging(bool enable) { detail::PrefetchConfig::instance().debug = enable; }
}  // namespace cudf::experimental::prefetch
