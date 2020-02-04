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

#pragma once

#include <tests/binaryop/util/operation.h>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/cudf_gtest.hpp>

#include <cudf/scalar/scalar.hpp>

#include <limits>

namespace cudf {
namespace test {
namespace binop {

template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOp,
          typename ScalarType = cudf::experimental::scalar_type_t<TypeLhs>>
void ASSERT_BINOP(column_view const& out,
                  scalar const& lhs,
                  column_view const& rhs,
                  TypeOp&& op) {
    auto lhs_h = static_cast<ScalarType const&>(lhs).operator TypeLhs();
    auto rhs_h = cudf::test::to_host<TypeRhs>(rhs);
    auto rhs_data = rhs_h.first;
    auto out_h = cudf::test::to_host<TypeOut>(out);
    auto out_data = out_h.first;

    ASSERT_EQ(out_data.size(), rhs_data.size());
    for (size_t i = 0; i < out_data.size(); ++i) {
        ASSERT_EQ(out_data[i], (TypeOut)(op(lhs_h, rhs_data[i])));
    }

    if (rhs.nullable()) {
        ASSERT_TRUE(out.nullable());
        auto rhs_valid = rhs_h.second;
        auto out_valid = out_h.second;

        uint32_t lhs_valid = (lhs.is_valid() 
                            ? std::numeric_limits<bitmask_type>::max() : 0);
        ASSERT_EQ(out_valid.size(), rhs_valid.size());
        for (decltype(out_valid.size()) i = 0; i < out_valid.size(); ++i) {
            ASSERT_EQ(out_valid[i], (lhs_valid & rhs_valid[i]));
        }
    } 
    else {
        if (lhs.is_valid()) {
            ASSERT_FALSE(out.nullable());
        } 
        else {
            auto out_valid = out_h.second;
            for (decltype(out_valid.size()) i = 0; i < out_valid.size(); ++i) {
                ASSERT_EQ(out_valid[i], 0);
            }
        }
    }
}

template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOp,
          typename ScalarType = cudf::experimental::scalar_type_t<TypeRhs>>
void ASSERT_BINOP(column_view const& out,
                  column_view const& lhs,
                  scalar const& rhs,
                  TypeOp&& op) {
    auto rhs_h = static_cast<ScalarType const&>(rhs).operator TypeRhs();
    auto lhs_h = cudf::test::to_host<TypeLhs>(lhs);
    auto lhs_data = lhs_h.first;
    auto out_h = cudf::test::to_host<TypeOut>(out);
    auto out_data = out_h.first;

    ASSERT_EQ(out_data.size(), lhs_data.size());
    for (size_t i = 0; i < out_data.size(); ++i) {
        ASSERT_EQ(out_data[i], (TypeOut)(op(lhs_data[i], rhs_h)));
    }

    if (lhs.nullable()) {
        ASSERT_TRUE(out.nullable());
        auto lhs_valid = lhs_h.second;
        auto out_valid = out_h.second;

        uint32_t rhs_valid = (rhs.is_valid() 
                            ? std::numeric_limits<bitmask_type>::max() : 0);
        ASSERT_EQ(out_valid.size(), lhs_valid.size());
        for (decltype(out_valid.size()) i = 0; i < out_valid.size(); ++i) {
            ASSERT_EQ(out_valid[i], (rhs_valid & lhs_valid[i]));
        }
    } 
    else {
        if (rhs.is_valid()) {
            ASSERT_FALSE(out.nullable());
        } 
        else {
            auto out_valid = out_h.second;
            for (decltype(out_valid.size()) i = 0; i < out_valid.size(); ++i) {
                ASSERT_EQ(out_valid[i], 0);
            }
        }
    }
}

template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOp>
void ASSERT_BINOP(column_view const& out,
                  column_view const& lhs,
                  column_view const& rhs,
                  TypeOp&& op) {
    auto lhs_h = cudf::test::to_host<TypeLhs>(lhs);
    auto lhs_data = lhs_h.first;
    auto rhs_h = cudf::test::to_host<TypeRhs>(rhs);
    auto rhs_data = rhs_h.first;
    auto out_h = cudf::test::to_host<TypeOut>(out);
    auto out_data = out_h.first;

    ASSERT_EQ(out_data.size(), lhs_data.size());
    ASSERT_EQ(out_data.size(), rhs_data.size());
    for (size_t i = 0; i < out_data.size(); ++i) {
        ASSERT_EQ(out_data[i], (TypeOut)(op(lhs_data[i], rhs_data[i])));
    }

    if (lhs.nullable() and rhs.nullable()) {
        ASSERT_TRUE(out.nullable());
        auto lhs_valid = lhs_h.second;
        auto rhs_valid = rhs_h.second;
        auto out_valid = out_h.second;

        ASSERT_EQ(out_valid.size(), lhs_valid.size());
        ASSERT_EQ(out_valid.size(), rhs_valid.size());
        for (decltype(out_valid.size()) i = 0; i < out_valid.size(); ++i) {
            ASSERT_EQ(out_valid[i], (lhs_valid[i] & rhs_valid[i]));
        }
    }
    else if (not lhs.nullable() and rhs.nullable()) {
        ASSERT_TRUE(out.nullable());
        auto rhs_valid = rhs_h.second;
        auto out_valid = out_h.second;

        ASSERT_EQ(out_valid.size(), rhs_valid.size());
        for (decltype(out_valid.size()) i = 0; i < out_valid.size(); ++i) {
            ASSERT_EQ(out_valid[i], rhs_valid[i]);
        }
    }
    else if (lhs.nullable() and not rhs.nullable()) {
        ASSERT_TRUE(out.nullable());
        auto lhs_valid = lhs_h.second;
        auto out_valid = out_h.second;

        ASSERT_EQ(out_valid.size(), lhs_valid.size());
        for (decltype(out_valid.size()) i = 0; i < out_valid.size(); ++i) {
            ASSERT_EQ(out_valid[i], lhs_valid[i]);
        }
    }
    else {
        ASSERT_FALSE(out.nullable());
    }
}

/**
 * According to CUDA Programming Guide, 'E.1. Standard Functions', 'Table 7 - Double-Precision
 * Mathematical Standard Library Functions with Maximum ULP Error'
 * The pow function has 2 (full range) maximum ulp error.
 */
template <typename TypeOut, typename TypeLhs, typename TypeRhs>
void ASSERT_BINOP(column_view const& out,
                  column_view const& lhs,
                  column_view const& rhs,
                  cudf::library::operation::Pow<TypeOut, TypeLhs, TypeRhs>&& op) {
    auto lhs_h = cudf::test::to_host<TypeLhs>(lhs);
    auto lhs_data = lhs_h.first;
    auto rhs_h = cudf::test::to_host<TypeRhs>(rhs);
    auto rhs_data = rhs_h.first;
    auto out_h = cudf::test::to_host<TypeOut>(out);
    auto out_data = out_h.first;

    const int ULP = 2.0;
    ASSERT_EQ(out_data.size(), lhs_data.size());
    ASSERT_EQ(out_data.size(), rhs_data.size());
    for (decltype(out_data.size()) index = 0; index < out_data.size(); ++index) {
        ASSERT_TRUE(abs(out_data[index] - (TypeOut)(op(lhs_data[index], rhs_data[index]))) < ULP);
    }

    auto lhs_valid = lhs_h.second;
    auto rhs_valid = rhs_h.second;
    auto out_valid = out_h.second;

    ASSERT_EQ(out_valid.size(), lhs_valid.size());
    ASSERT_EQ(out_valid.size(), rhs_valid.size());
    for (decltype(out_valid.size()) index = 0; index < out_valid.size(); ++index) {
        ASSERT_EQ(out_valid[index], (lhs_valid[index] & rhs_valid[index]));
    }
}

} // namespace binop
} // namespace test
} // namespace cudf
