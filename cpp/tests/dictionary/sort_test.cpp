/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>
#include <thrust/sequence.h>

struct DictionarySortTest : public cudf::test::BaseFixture {};

TEST_F(DictionarySortTest, StringColumn)
{
    std::vector<const char*> h_strings{ "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );

    auto dictionary = cudf::dictionary::encode( strings );
    cudf::dictionary_column_view view(dictionary->view());

    cudf::table_view input{{strings}};
    std::vector<cudf::order> column_order{cudf::order::ASCENDING};
    auto got = cudf::experimental::sorted_order(cudf::table_view{{dictionary->view()}},
                                                std::vector<cudf::order>{cudf::order::ASCENDING});

    auto h_indices = cudf::test::to_host<int32_t>(view.indices()).first;
    std::vector<int32_t> h_expected(got->size());
    thrust::sequence( h_expected.begin(), h_expected.end() );
    std::sort( h_expected.begin(), h_expected.end(), [h_indices] (int32_t lhs, int32_t rhs) { return h_indices[lhs] < h_indices[rhs]; } );
    cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected.begin(), h_expected.end() );
    cudf::test::expect_columns_equal(got->view(), expected);
}
