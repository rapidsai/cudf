/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <groupby/common/utils.hpp>

#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf::groupby::detail {

void assert_keys_equality_comparable(cudf::table_view const& keys)
{
  auto is_supported_key_type = [](auto col) { return cudf::is_equality_comparable(col.type()); };
  CUDF_EXPECTS(std::all_of(keys.begin(), keys.end(), is_supported_key_type),
               "Unsupported groupby key type does not support equality comparison");
}

}  // namespace cudf::groupby::detail
