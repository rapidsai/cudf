/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <tests/iterator/pair_iterator_test.cuh>

using TestingTypes = cudf::test::ChronoTypes;

template <typename T>
struct ChronoPairIteratorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(ChronoPairIteratorTest, TestingTypes);
TYPED_TEST(ChronoPairIteratorTest, nonull_pair_iterator) { nonull_pair_iterator(*this); }
TYPED_TEST(ChronoPairIteratorTest, null_pair_iterator) { null_pair_iterator(*this); }
