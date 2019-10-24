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
#include <cudf/utilities/nvtx_utils.hpp>
#include "cudf/utilities/error.hpp"

#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

namespace cudf {
namespace nvtx {

void range_push(const char* name, color color)
{
  range_push_hex(name, static_cast<uint32_t>(color));
}

void range_push_hex(const char* name, uint32_t color)
{
#ifdef USE_NVTX
  CUDF_EXPECTS(name != nullptr, "Null name string.");

  nvtxEventAttributes_t eventAttrib{};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name;
  nvtxRangePushEx(&eventAttrib);
#endif
}

void range_pop()
{
#ifdef USE_NVTX
  nvtxRangePop();
#endif
}

}  // namespace nvtx
}  // namespace cudf
