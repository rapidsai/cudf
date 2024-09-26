/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "hash_compound_agg_finalizer.cuh"
#include "helpers.cuh"

namespace cudf::groupby::detail::hash {

template class hash_compound_agg_finalizer<hash_set_ref_t>;
template class hash_compound_agg_finalizer<nullable_hash_set_ref_t>;

}  // namespace cudf::groupby::detail::hash
