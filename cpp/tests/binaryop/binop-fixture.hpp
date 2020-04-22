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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace test {
namespace binop {

struct BinaryOperationTest
    : public cudf::test::BaseFixture
    , cudf::test::UniformRandomGenerator<uint8_t>
{
    BinaryOperationTest()
     : cudf::test::UniformRandomGenerator<uint8_t>(1, 10)
    {}

    auto make_data_iter() {
        return cudf::test::make_counting_transform_iterator(0,
            [this](auto row) { return this->generate(); });
    }

    auto make_validity_iter() {
        uint8_t mod_base = this->generate();
        return cudf::test::make_counting_transform_iterator(0,
            [mod_base](auto row) { return (row % mod_base) > 0; });
    }

    template <typename T>
    auto make_random_wrapped_column(size_type size) {
        auto data_iter = make_data_iter();
        auto validity_iter = make_validity_iter();
        return cudf::test::fixed_width_column_wrapper<T>(
            data_iter, data_iter + size, validity_iter);
    }

    template <typename T>
    auto make_random_wrapped_scalar() {
        return cudf::experimental::scalar_type_t<T>(this->generate());
    }
};

} // namespace binop
} // namespace test
} // namespace cudf
