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

#pragma once

#include <nvtx3/nvtx3.hpp>

namespace cudf {
/**
 * @brief Tag type for libcudf's NVTX domain.
 */
struct libcudf_domain {
  static constexpr char const* name{"libcudf"};  ///< Name of the libcudf domain
};

/**
 * @brief Alias for an NVTX range in the libcudf domain.
 *
 * Customizes an NVTX range with the given input.
 *
 * Example:
 * ```
 * void some_function(){
 *    cudf::scoped_range rng{"custom_name"}; // Customizes range name
 *    ...
 * }
 * ```
 */
using scoped_range = ::nvtx3::scoped_range_in<libcudf_domain>;

namespace detail {
constexpr uint32_t cudf_nvtx_default_color{0xffbf00};

/**
 * @brief
 *
 */
class nvtx_event_attr {
 public:
  nvtx_event_attr(const nvtxStringHandle_t string_handle, uint32_t color)
  {
    std::memset(&_attr, 0, NVTX_EVENT_ATTRIB_STRUCT_SIZE);
    _attr.version            = NVTX_VERSION;
    _attr.size               = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    _attr.colorType          = nvtxColorType_t::NVTX_COLOR_ARGB;
    _attr.color              = color;
    _attr.messageType        = nvtxMessageType_t::NVTX_MESSAGE_TYPE_REGISTERED;
    _attr.message.registered = string_handle;
  }

  const nvtxEventAttributes_t& get() const { return _attr; }

 private:
  nvtxEventAttributes_t _attr;
};

class nvtx_scoped_range {
 public:
  nvtx_scoped_range(const nvtx_event_attr& event_attr, nvtxDomainHandle_t domain) : _domain(domain)
  {
    nvtxDomainRangePushEx(_domain, &event_attr.get());
  }

  ~nvtx_scoped_range() { nvtxDomainRangePop(_domain); }

 private:
  nvtxDomainHandle_t _domain{};
};
}  // namespace detail

}  // namespace cudf

/**
 * @brief Convenience macro for generating an NVTX range in the `libcudf` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    CUDF_FUNC_RANGE();
 *    ...
 * }
 * ```
 */
#define CUDF_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(cudf::libcudf_domain)

/**
 * @brief
 *
 */
#define GET_EVENT_ATTR(range_name, color)                                                     \
  [](const char* range_name_var, uint32_t color_var) -> cudf::detail::nvtx_event_attr& {      \
    static auto* string_handle =                                                              \
      nvtxDomainRegisterString(::nvtx3::domain::get<cudf::libcudf_domain>(), range_name_var); \
    static cudf::detail::nvtx_event_attr event_attr(string_handle, color_var);                \
    return event_attr;                                                                        \
  }(range_name, color)

/**
 * @brief
 *
 */
#define CUDF_RANGE_PUSH(range_name)                                   \
  nvtxDomainRangePushEx(::nvtx3::domain::get<cudf::libcudf_domain>(), \
                        &GET_EVENT_ATTR(range_name, cudf::detail::cudf_nvtx_default_color).get());

/**
 * @brief
 *
 */
#define CUDF_RANGE_PUSH_COLOR(range_name, color)                      \
  nvtxDomainRangePushEx(::nvtx3::domain::get<cudf::libcudf_domain>(), \
                        &GET_EVENT_ATTR(range_name, color).get());

/**
 * @brief
 *
 */
#define CUDF_RANGE_POP() nvtxDomainRangePop(::nvtx3::domain::get<cudf::libcudf_domain>());

/**
 * @brief
 *
 */
#define CONCAT2(x, y)   x##y
#define CONCAT(x, y)    CONCAT2(x, y)
#define UNIQUE_VAR(var) CONCAT(var, __LINE__)

/**
 * @brief
 *
 */
// clang-format off
#define CUDF_SCOPED_RANGE(range_name) cudf::detail::nvtx_scoped_range UNIQUE_VAR(nvtx_scoped_range_var)(GET_EVENT_ATTR(range_name, cudf::detail::cudf_nvtx_default_color), ::nvtx3::domain::get<cudf::libcudf_domain>());
// clang-format on