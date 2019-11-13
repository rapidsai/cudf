/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

 #include <cudf/copying.hpp>
 #include <cudf/table/table.hpp>
 #include <cudf/table/table_view.hpp>
 #include <cudf/scalar/scalar.hpp>

 namespace cudf {

 std::unique_ptr<table> shift(const table_view& in,
                              size_type periods,
                              const std::vector<scalar>& fill_value = {},
                              rmm::mr::device_memory_resource *mr =
                                  rmm::mr::get_default_resource())
{
    // if (periods == 0 || in.num_rows() == 0) {
    //     // return cudf::experimental::empty_like(in);
    //     return table(in);
    // }

    if (not fill_value.empty()) {
        CUDF_EXPECTS(static_cast <unsigned int>(in.num_columns()) == fill_value.size(), "`fill_value.size()` and `in.num_columns() must be the same.");
    }

    throw cudf::logic_error("not implemented");
}

} // namespace cudf
