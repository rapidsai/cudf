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

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeVal, typename TypeOpe>
void ASSERT_BINOP(gdf::library::Vector<TypeOut>& out,
                  gdf::library::Scalar<TypeVax>& vax,
                  gdf::library::Vector<TypeVay>& vay,
                  gdf::library::Scalar<TypeVal>& def,
                  TypeOpe&& ope) {
    using ValidType = typename gdf::library::Vector<TypeOut>::ValidType;
    int ValidSize = gdf::library::Vector<TypeOut>::ValidSize;

    ASSERT_TRUE(out.dataSize() == vay.dataSize());
    ASSERT_TRUE(out.validSize() == vay.validSize());

    ValidType mask = 1;
    int index_valid = 0;
    for (int index = 0; index < out.dataSize(); ++index) {
        if (!(index % ValidSize)) {
            mask = 1;
            index_valid = index / ValidSize;
        } else {
            mask <<= 1;
        }

        TypeVay vax_aux = (TypeVay)vax;
        if (!vax.isValid()) {
            vax_aux = (TypeVay)((TypeVal) def.getValue());
        }

        TypeVax vay_aux = vay.data[index];
        if ((vay.valid[index_valid] & mask) == 0) {
            vay_aux = (TypeVal) def.getValue();
        }

        ASSERT_TRUE(out.data[index] == (TypeOut)(ope(vax_aux, vay_aux)));
    }

    uint32_t vax_valid = (vax.isValid() ? UINT32_MAX : 0);
    uint32_t def_valid = (def.isValid() ? UINT32_MAX : 0);
    ASSERT_TRUE(out.validSize() == vay.validSize());
    for (int index = 0; index < out.validSize(); ++index) {
        uint32_t output = (vay.valid[index] & vax_valid) |
                          (vay.valid[index] & def_valid) |
                          (vax_valid & def_valid);
        ASSERT_TRUE(out.valid[index] == output);
    }
}

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeVal, typename TypeOpe>
void ASSERT_BINOP(gdf::library::Vector<TypeOut>& out,
                  gdf::library::Vector<TypeVax>& vax,
                  gdf::library::Scalar<TypeVay>& vay,
                  gdf::library::Scalar<TypeVal>& def,
                  TypeOpe&& ope) {
    using ValidType = typename gdf::library::Vector<TypeOut>::ValidType;
    int ValidSize = gdf::library::Vector<TypeOut>::ValidSize;

    ASSERT_TRUE(out.dataSize() == vax.dataSize());
    ASSERT_TRUE(out.validSize() == vax.validSize());

    ValidType mask = 1;
    int index_valid = 0;
    for (int index = 0; index < out.dataSize(); ++index) {
        if (!(index % ValidSize)) {
            mask = 1;
            index_valid = index / ValidSize;
        } else {
            mask <<= 1;
        }

        TypeVax vax_aux = vax.data[index];
        if ((vax.valid[index_valid] & mask) == 0) {
            vax_aux = (TypeVal) def.getValue();
        }

        TypeVay vay_aux = (TypeVay)vay;
        if (!vay.isValid()) {
            vay_aux = (TypeVay)((TypeVal) def.getValue());
        }

        ASSERT_TRUE(out.data[index] == (TypeOut)(ope(vax_aux, vay_aux)));
    }

    uint32_t vay_valid = (vay.isValid() ? UINT32_MAX : 0);
    uint32_t def_valid = (def.isValid() ? UINT32_MAX : 0);
    ASSERT_TRUE(out.validSize() == vax.validSize());
    for (int index = 0; index < out.validSize(); ++index) {
        uint32_t output = (vax.valid[index] & vay_valid) |
                          (vax.valid[index] & def_valid) |
                          (vay_valid & def_valid);
        ASSERT_TRUE(out.valid[index] == output);
    }
}

template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeDef, typename TypeOpe>
void ASSERT_BINOP(gdf::library::Vector<TypeOut>& out,
                  gdf::library::Vector<TypeVax>& vax,
                  gdf::library::Vector<TypeVay>& vay,
                  gdf::library::Scalar<TypeDef>& def,
                  TypeOpe&& ope) {
    using ValidType = typename gdf::library::Vector<TypeOut>::ValidType;
    int ValidSize = gdf::library::Vector<TypeOut>::ValidSize;

    ASSERT_TRUE(out.dataSize() == vax.dataSize());
    ASSERT_TRUE(out.dataSize() == vay.dataSize());

    ValidType mask = 1;
    int index_valid = 0;
    for (int index = 0; index < out.dataSize(); ++index) {
        if (!(index % ValidSize)) {
            mask = 1;
            index_valid = index / ValidSize;
        } else {
            mask <<= 1;
        }

        TypeVax vax_aux = vax.data[index];
        if ((vax.valid[index_valid] & mask) == 0) {
            vax_aux = (TypeVax) def.getValue();
        }

        TypeVay vay_aux = vay.data[index];
        if ((vay.valid[index_valid] & mask) == 0) {
            vay_aux = (TypeVay) def.getValue();
        }

        ASSERT_TRUE(out.data[index] == (TypeOut)(ope(vax_aux, vay_aux)));
    }

    uint32_t def_valid = (def.isValid() ? UINT32_MAX : 0);
    ASSERT_TRUE(out.validSize() == vax.validSize());
    ASSERT_TRUE(out.validSize() == vay.validSize());
    for (int index = 0; index < out.validSize(); ++index) {
        ASSERT_TRUE(out.valid[index] == ((vax.valid[index] & vay.valid[index]) |
                                         (vax.valid[index] & def_valid) |
                                         (vay.valid[index] & def_valid)));
    }
}

} // namespace binop
} // namespace test
} // namespace gdf

#endif
