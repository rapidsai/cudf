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

#include "gtest/gtest.h"
#include "tests/binary/util/scalar.h"
#include "tests/binary/util/vector.h"
#include "tests/binary/util/operation.h"

namespace gdf {
namespace test {
namespace binop {

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
void ASSERT_BINOP(gdf::library::Vector<TypeOut>& out,
                  gdf::library::Scalar<TypeVax>& vax,
                  gdf::library::Vector<TypeVay>& vay,
                  TypeOpe&& ope) {
    ASSERT_TRUE(out.dataSize() == vay.dataSize());
    for (int index = 0; index < out.dataSize(); ++index) {
        ASSERT_TRUE(out.data[index] == (TypeOut)(ope((TypeVay) vax.getValue(), vay.data[index])));
    }

    uint32_t vax_valid = (vax.isValid() ? UINT32_MAX : 0);
    ASSERT_TRUE(out.validSize() == vay.validSize());
    for (int index = 0; index < out.validSize(); ++index) {
        ASSERT_TRUE(out.valid[index] == (vax_valid & vay.valid[index]));
    }
}

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
void ASSERT_BINOP(gdf::library::Vector<TypeOut>& out,
                  gdf::library::Vector<TypeVax>& vax,
                  gdf::library::Scalar<TypeVay>& vay,
                  TypeOpe&& ope) {
    ASSERT_TRUE(out.dataSize() == vax.dataSize());
    for (int index = 0; index < out.dataSize(); ++index) {
        ASSERT_TRUE(out.data[index] == (TypeOut)(ope(vax.data[index], (TypeVay) vay.getValue())));
    }

    uint32_t vay_valid = (vay.isValid() ? UINT32_MAX : 0);
    ASSERT_TRUE(out.validSize() == vax.validSize());
    for (int index = 0; index < out.validSize(); ++index) {
        ASSERT_TRUE(out.valid[index] == (vax.valid[index] & vay_valid));
    }
}

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
void ASSERT_BINOP(gdf::library::Vector<TypeOut>& out,
                  gdf::library::Vector<TypeVax>& vax,
                  gdf::library::Vector<TypeVay>& vay,
                  TypeOpe&& ope) {
    ASSERT_TRUE(out.dataSize() == vax.dataSize());
    ASSERT_TRUE(out.dataSize() == vay.dataSize());
    for (int index = 0; index < out.dataSize(); ++index) {
        ASSERT_TRUE(out.data[index] == (TypeOut)(ope(vax.data[index], vay.data[index])));
    }

    ASSERT_TRUE(out.validSize() == vax.validSize());
    ASSERT_TRUE(out.validSize() == vay.validSize());
    for (int index = 0; index < out.validSize(); ++index) {
        ASSERT_TRUE(out.valid[index] == vax.valid[index] | vay.valid[index]);
    }
}

/**
 * According to CUDA Programming Guide, 'E.1. Standard Functions', 'Table 7 - Double-Precision
 * Mathematical Standard Library Functions with Maximum ULP Error'
 * The pow function has 2 (full range) maximum ulp error.
 */
template <typename TypeOut, typename TypeVax, typename TypeVay>
void ASSERT_BINOP(gdf::library::Vector<TypeOut>& out,
                  gdf::library::Vector<TypeVax>& vax,
                  gdf::library::Vector<TypeVay>& vay,
                  gdf::library::operation::Pow<TypeOut, TypeVax, TypeVay>&& ope) {
    const int ULP = 2.0;
    ASSERT_TRUE(out.dataSize() == vax.dataSize());
    ASSERT_TRUE(out.dataSize() == vay.dataSize());
    for (int index = 0; index < out.dataSize(); ++index) {
        ASSERT_TRUE(abs(out.data[index] - (TypeOut)(ope(vax.data[index], vay.data[index]))) < ULP);
    }

    ASSERT_TRUE(out.validSize() == vax.validSize());
    ASSERT_TRUE(out.validSize() == vay.validSize());
    for (int index = 0; index < out.validSize(); ++index) {
        ASSERT_TRUE(out.valid[index] == (vax.valid[index] & vay.valid[index]));
    }
}

} // namespace binop
} // namespace test
} // namespace gdf

#endif
