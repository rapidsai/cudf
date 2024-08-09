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

#include <benchmarks/stream_compaction/stream_compaction_common.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/error.hpp>

cudf::duplicate_keep_option get_keep(std::string const& keep_str)
{
  if (keep_str == "any") {
    return cudf::duplicate_keep_option::KEEP_ANY;
  } else if (keep_str == "first") {
    return cudf::duplicate_keep_option::KEEP_FIRST;
  } else if (keep_str == "last") {
    return cudf::duplicate_keep_option::KEEP_LAST;
  } else if (keep_str == "none") {
    return cudf::duplicate_keep_option::KEEP_NONE;
  } else {
    CUDF_FAIL("Unsupported keep option.");
  }
}
