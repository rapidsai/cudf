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

#include <cudf/types.hpp>

namespace cudf {
namespace test {
namespace detail {

/**
 * @copydoc cudf::test::detail::random_generator_incrementing_seed()
 */
uint64_t random_generator_incrementing_seed()
{
  static uint64_t seed = 0;
  return ++seed;
}

}  // namespace detail
}  // namespace test
}  // namespace cudf
