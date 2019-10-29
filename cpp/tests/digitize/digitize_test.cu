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
#include <cudf/digitize.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

// TODO clean this up and move it somewhere
#include <cuda_runtime_api.h>
void print_column(cudf::column_view view) {
    std::vector<int32_t> vec(view.size());
    cudaMemcpy(vec.data(), view.head(), sizeof(int32_t) * view.size(), cudaMemcpyDefault);
    for (auto const& val : vec) {
        std::cout << val << ", ";
    }
    std::cout << std::endl;
}

template <typename T>
struct Digitize : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(Digitize, cudf::test::NumericTypes);

TYPED_TEST(Digitize, WITH_INCLUSIVE)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> bins{{0, 4, 8}, {1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<T> values{{-2, 0, 2, 4, 6, 8, 10}, {1, 1, 1, 1, 1, 1, 1}};

    cudf::test::fixed_width_column_wrapper<int32_t> expected{{0, 0, 1, 1, 2, 2, 3}};

    auto out = cudf::digitize(values, bins, cudf::range_bound::INCLUSIVE, cudf::null_order::AFTER);
    print_column(out->view());

    cudf::test::expect_columns_equal(expected, out->view());
}

TYPED_TEST(Digitize, WITH_EXCLUSIVE)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> bins{{0, 4, 8}, {1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<T> values{{-2, 0, 2, 4, 6, 8, 10}, {1, 1, 1, 1, 1, 1, 1}};

    cudf::test::fixed_width_column_wrapper<int32_t> expected{{0, 1, 1, 2, 2, 3, 3}};

    auto out = cudf::digitize(values, bins, cudf::range_bound::EXCLUSIVE, cudf::null_order::AFTER);
    print_column(out->view());

    cudf::test::expect_columns_equal(expected, out->view());
}
