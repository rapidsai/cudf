/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <memory>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include "cudf/detail/gather.hpp"
#include "thrust/gather.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/transform.h"

namespace cudf {
namespace experimental {

std::unique_ptr<table> shift(table_view const& in,
    size_type periods,
    std::vector<scalar> const& fill_value)
{
    if (in.num_rows() == 0) {
        return empty_like(in);
    }

    throw new cudf::logic_error("not implemented");
}

} // namespace experimental
} // namespace cudf
