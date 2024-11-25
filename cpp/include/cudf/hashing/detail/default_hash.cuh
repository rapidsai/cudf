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

#pragma once

#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>

namespace cudf::hashing::detail {

/**
 * @brief The default hash algorithm for use within libcudf internal functions
 *
 * This is declared here so it may be changed to another algorithm without modifying
 * all those places that use it. Internal function implementations are encourage to
 * use the `cudf::hashing::detail::default_hash` where possible.
 *
 * @tparam Key The key type for use by the hash class
 */
template <typename Key>
using default_hash = MurmurHash3_x86_32<Key>;

}  // namespace cudf::hashing::detail
