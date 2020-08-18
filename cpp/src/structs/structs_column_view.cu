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

#include <cudf/column/column.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include "cudf/utilities/error.hpp"

namespace cudf {

structs_column_view::structs_column_view(column_view const& rhs) : column_view{rhs}
{
  CUDF_EXPECTS(type().id() == type_id::STRUCT, "structs_column_view only supports struct columns");
}

}  // namespace cudf