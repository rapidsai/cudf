/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS,  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/type_lists.hpp>

#include <tests/iterator/value_iterator_test.cuh>

using TestingTypes = cudf::test::NumericTypes;

template <typename T>
struct NumericValueIteratorTest : public IteratorTest<T> {
};

TYPED_TEST_CASE(NumericValueIteratorTest, TestingTypes);
TYPED_TEST(NumericValueIteratorTest, non_null_iterator) { non_null_iterator(*this); }
TYPED_TEST(NumericValueIteratorTest, null_iterator) { null_iterator(*this); }
