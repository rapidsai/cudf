/*
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

#ifndef GDF_TESTS_BINARY_OPERATION_INTEGRATION_ASSERT_BINOPS_H
#define GDF_TESTS_BINARY_OPERATION_INTEGRATION_ASSERT_BINOPS_H

#include "tests/binary/util/operation.h"
#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/scalar_wrapper.cuh"
#include "gtest/gtest.h"

namespace gdf {
namespace test {
namespace binop {

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
void ASSERT_BINOP(cudf::test::column_wrapper<TypeOut>& out,
                  cudf::test::scalar_wrapper<TypeVax>& vax,
                  cudf::test::column_wrapper<TypeVay>& vay,
                  TypeOpe&& ope) {
    auto vax_h = vax.to_host();
    auto vay_h = vay.to_host();
    auto vay_data = std::get<0>(vay_h);
    auto out_h = out.to_host();
    auto out_data = std::get<0>(out_h);

    ASSERT_TRUE(out_data.size() == vay_data.size());
    for (int index = 0; index < out_data.size(); ++index) {
        ASSERT_TRUE(out_data[index] == (TypeOut)(ope(vax_h, vay_data[index])));
    }

    auto vay_valid = std::get<1>(vay_h);
    auto out_valid = std::get<1>(out_h);

    ASSERT_TRUE(out_valid.size() == vay_valid.size());
    for (int index = 0; index < out_valid.size(); ++index) {
        ASSERT_TRUE(out_valid[index] == vay_valid[index]);
    }
}

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
void ASSERT_BINOP(cudf::test::column_wrapper<TypeOut>& out,
                  cudf::test::column_wrapper<TypeVax>& vax,
                  cudf::test::scalar_wrapper<TypeVay>& vay,
                  TypeOpe&& ope) {
    auto vay_h = vay.to_host();
    auto vax_h = vax.to_host();
    auto vax_data = std::get<0>(vax_h);
    auto out_h = out.to_host();
    auto out_data = std::get<0>(out_h);

    ASSERT_TRUE(out_data.size() == vax_data.size());
    for (int index = 0; index < out_data.size(); ++index) {
        ASSERT_TRUE(out_data[index] == (TypeOut)(ope(vax_data[index], vay_h)));
    }

    auto vax_valid = std::get<1>(vax_h);
    auto out_valid = std::get<1>(out_h);

    ASSERT_TRUE(out_valid.size() == vax_valid.size());
    for (int index = 0; index < out_valid.size(); ++index) {
        ASSERT_TRUE(out_valid[index] == (vax_valid[index]));
    }
}

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
void ASSERT_BINOP(cudf::test::column_wrapper<TypeOut>& out,
                  cudf::test::column_wrapper<TypeVax>& vax,
                  cudf::test::column_wrapper<TypeVay>& vay,
                  TypeOpe&& ope) {
    auto vax_h = vax.to_host();
    auto vax_data = std::get<0>(vax_h);
    auto vay_h = vay.to_host();
    auto vay_data = std::get<0>(vay_h);
    auto out_h = out.to_host();
    auto out_data = std::get<0>(out_h);

    ASSERT_TRUE(out_data.size() == vax_data.size());
    ASSERT_TRUE(out_data.size() == vay_data.size());
    for (int index = 0; index < out_data.size(); ++index) {
        ASSERT_TRUE(out_data[index] == (TypeOut)(ope(vax_data[index], vay_data[index])));
    }

    auto vax_valid = std::get<1>(vax_h);
    auto vay_valid = std::get<1>(vay_h);
    auto out_valid = std::get<1>(out_h);

    ASSERT_TRUE(out_valid.size() == vax_valid.size());
    ASSERT_TRUE(out_valid.size() == vay_valid.size());
    for (int index = 0; index < out_valid.size(); ++index) {
        ASSERT_TRUE(out_valid[index] == vax_valid[index] | vay_valid[index]);
    }
}

/**
 * According to CUDA Programming Guide, 'E.1. Standard Functions', 'Table 7 - Double-Precision
 * Mathematical Standard Library Functions with Maximum ULP Error'
 * The pow function has 2 (full range) maximum ulp error.
 */
template <typename TypeOut, typename TypeVax, typename TypeVay>
void ASSERT_BINOP(cudf::test::column_wrapper<TypeOut>& out,
                  cudf::test::column_wrapper<TypeVax>& vax,
                  cudf::test::column_wrapper<TypeVay>& vay,
                  gdf::library::operation::Pow<TypeOut, TypeVax, TypeVay>&& ope) {
    auto vax_h = vax.to_host();
    auto vax_data = std::get<0>(vax_h);
    auto vay_h = vay.to_host();
    auto vay_data = std::get<0>(vay_h);
    auto out_h = out.to_host();
    auto out_data = std::get<0>(out_h);

    const int ULP = 2.0;
    ASSERT_TRUE(out_data.size() == vax_data.size());
    ASSERT_TRUE(out_data.size() == vay_data.size());
    for (int index = 0; index < out_data.size(); ++index) {
        ASSERT_TRUE(abs(out_data[index] - (TypeOut)(ope(vax_data[index], vay_data[index]))) < ULP);
    }

    auto vax_valid = std::get<1>(vax_h);
    auto vay_valid = std::get<1>(vay_h);
    auto out_valid = std::get<1>(out_h);

    ASSERT_TRUE(out_valid.size() == vax_valid.size());
    ASSERT_TRUE(out_valid.size() == vay_valid.size());
    for (int index = 0; index < out_valid.size(); ++index) {
        ASSERT_TRUE(out_valid[index] == (vax_valid[index] & vay_valid[index]));
    }
}

} // namespace binop
} // namespace test
} // namespace gdf

#endif
