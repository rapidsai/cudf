/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

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

CUDF_TEST_PROGRAM_MAIN()
