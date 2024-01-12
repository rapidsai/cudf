/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "reader_impl.hpp"

namespace cudf::io::orc::detail {

void reader::impl::compute_chunk_ranges()
{
  // Currently, file is always read as one chunk.
  // TODO: Implement variable chunking.
  _chunk_read_info.chunk_ranges = {
    row_range{_file_itm_data.rows_to_skip, _file_itm_data.rows_to_read}};
}

}  // namespace cudf::io::detail::orc
