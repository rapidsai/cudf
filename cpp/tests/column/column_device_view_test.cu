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

#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/cudf_gtest.hpp>

struct ColumnDeviceViewTest : public cudf::test::BaseFixture {};

TEST_F(ColumnDeviceViewTest, Sample) {
    using T = int32_t;
    cudaStream_t stream = 0;
    cudf::test::fixed_width_column_wrapper<T> input ({1, 2, 3, 4, 5, 6});
    auto output = cudf::experimental::allocate_like(input);
    auto input_device_view  = cudf::column_device_view::create(input, stream);
    auto output_device_view  = cudf::mutable_column_device_view::create(output->mutable_view(), stream);
    auto exec = rmm::exec_policy(stream);
    EXPECT_NO_THROW(thrust::copy(exec->on(stream), input_device_view->begin<T>(), input_device_view->end<T>(),
            output_device_view->begin<T>()));

    cudf::test::expect_columns_equal(input, output->view());
}

TEST_F(ColumnDeviceViewTest, MismatchingType) {
    using T = int32_t;
    cudaStream_t stream = 0;
    cudf::test::fixed_width_column_wrapper<T> input ({1, 2, 3, 4, 5, 6});
    auto output = cudf::experimental::allocate_like(input);
    auto input_device_view  = cudf::column_device_view::create(input, stream);
    auto output_device_view  = cudf::mutable_column_device_view::create(output->mutable_view(), stream);
    auto exec = rmm::exec_policy(stream);
    EXPECT_THROW(thrust::copy(exec->on(stream), input_device_view->begin<T>(),
                 input_device_view->end<T>(), output_device_view->begin<int64_t>()), cudf::logic_error);
}
