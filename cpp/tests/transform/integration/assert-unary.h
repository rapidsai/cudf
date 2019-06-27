/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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
#ifndef GDF_TESTS_UNARY_OPERATION_INTEGRATION_ASSERT_BINOPS_H
#define GDF_TESTS_UNARY_OPERATION_INTEGRATION_ASSERT_BINOPS_H

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/scalar_wrapper.cuh>
#include <gtest/gtest.h>

namespace cudf {
namespace test {
namespace transformation {

template <typename TypeOut, typename TypeIn, typename TypeOpe>
void ASSERT_UNARY(cudf::test::column_wrapper<TypeOut>& out,
                  cudf::test::column_wrapper<TypeIn>& in,
                  TypeOpe&& ope) {
    auto in_h = in.to_host();
    auto in_data = std::get<0>(in_h);
    auto out_h = out.to_host();
    auto out_data = std::get<0>(out_h);

    ASSERT_TRUE(out_data.size() == in_data.size());
    for (gdf_index_type index = 0; index < (gdf_index_type)out_data.size(); ++index) {
      EXPECT_EQ(out_data[index], (TypeOut)(ope(in_data[index])));
    }

    auto in_valid = std::get<1>(in_h);
    auto out_valid = std::get<1>(out_h);

    ASSERT_TRUE(out_valid.size() == in_valid.size());
    for (gdf_index_type index = 0; index < (gdf_index_type)out_valid.size(); ++index) {
        EXPECT_EQ(out_valid[index], in_valid[index]);
    }
}

}
}
}

#endif
