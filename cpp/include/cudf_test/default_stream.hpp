/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace test {

/**
 * @brief Get the default stream to use for tests.
 *
 * The standard behavior of this function is to return cudf's default stream
 * (cudf::get_default_stream). This function is primarily provided as an
 * overload target for preload libraries (via LD_PRELOAD) so that the default
 * stream used for tests may be modified for tracking purposes. All tests of
 * public APIs that accept streams should pass `cudf::test::get_default_stream`
 * as the stream argument so that a preload library changing the behavior of
 * this function will trigger those tests to run on a different stream than
 * `cudf::get_default_stream`.
 *
 * @return The default stream to use for tests.
 */
rmm::cuda_stream_view const get_default_stream();

}  // namespace test
}  // namespace CUDF_EXPORT cudf
