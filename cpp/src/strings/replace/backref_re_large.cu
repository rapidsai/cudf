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

#include "backref_re.cuh"

#include <cudf/strings/detail/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

//
children_pair replace_with_backrefs_large(column_device_view const& d_strings,
                                          reprog_device& d_prog,
                                          string_view const& d_repl_template,
                                          rmm::device_vector<backref_type>& backrefs,
                                          size_type null_count,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  return make_strings_children(
    backrefs_fn<RX_STACK_LARGE>{
      d_strings, d_prog, d_repl_template, backrefs.begin(), backrefs.end()},
    d_strings.size(),
    null_count,
    stream,
    mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
