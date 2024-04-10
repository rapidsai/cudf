/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/cuda_stream_view.hpp>

class StreamPoolTest : public cudf::test::BaseFixture {};

CUDF_KERNEL void do_nothing_kernel() {}

TEST_F(StreamPoolTest, ForkStreams)
{
  auto streams = cudf::detail::fork_streams(cudf::test::get_default_stream(), 2);
  for (auto& stream : streams) {
    do_nothing_kernel<<<1, 32, 0, stream.value()>>>();
  }
}
