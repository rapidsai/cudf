/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/dedup.hpp>

#include <vector>

struct TextDedupTest : public cudf::test::BaseFixture {};

TEST_F(TextDedupTest, StringDedup)
{
  // https://loremipsum.io/generator?n=25&t=p
  // clang-format off
  auto input = cudf::test::strings_column_wrapper({
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ", //  90
    "01234567890123456789 magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation     ", // 180
    "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit   ", // 270
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. 01234567890123456789         ", // 360
    "cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.    ", // 450
    "Ea esse numquam et recusandae quia et voluptatem sint quo explicabo repudiandae. At nihil ", // 540
    "sunt non architecto doloremque eos dolorem consequuntur. Vel adipisci quod et voluptatum  ", // 630
    "quis est fuga tempore qui dignissimos aliquam et sint repellendus ut autem voluptas quo   ", // 720
    "deleniti earum? Qui ipsam ipsum hic ratione mollitia aut nobis laboriosam. Eum aspernatur ", // 810
    "dolorem sit voluptatum numquam in iure placeat vel laudantium molestiae? Ad reprehenderit ", // 900
    "quia aut minima deleniti id consequatur sapiente est dolores cupiditate. 012345678901234  ", // 990
  });
  // clang-format on

  auto sv = cudf::strings_column_view(input);

  // auto results  = nvtext::substring_deduplicate(sv, 20);
  // auto expected = cudf::test::strings_column_wrapper({" 01234567890123456789 "});
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  auto results = nvtext::substring_deduplicate(sv, 15);
  cudf::test::print(results->view());
  // auto expected = cudf::test::strings_column_wrapper(
  //   {" 01234567890123456789 ", ". 012345678901234", " reprehenderit "});
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  // Test with sliced input
  //  auto const sliced_input = cudf::slice(input, {1, 10}).front();
  //
  //  sv       = cudf::strings_column_view(sliced_input);
  //  results  = nvtext::substring_deduplicate(sv, 15);
  //  expected = cudf::test::strings_column_wrapper({"01234567890123456789 ", " reprehenderit "});
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TEST_F(TextDedupTest, SuffixArray)
{
  // https://loremipsum.io/generator?n=25&t=p
  // clang-format off
  auto input = cudf::test::strings_column_wrapper({
    "Lorem ipsum dolor sit am"//et, consectetur adipiscing elit, sed do eiusmod tempor incididunt ", //  90
    // "01234567890123456789 magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation     ", // 180
    // "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit   ", // 270
    // "voluptate velit esse cillum dolore eu fugiat nulla pariatur. 01234567890123456789         ", // 360
    // "cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.    ", // 450
    // "Ea esse numquam et recusandae quia et voluptatem sint quo explicabo repudiandae. At nihil ", // 540
    // "sunt non architecto doloremque eos dolorem consequuntur. Vel adipisci quod et voluptatum  ", // 630
    // "quis est fuga tempore qui dignissimos aliquam et sint repellendus ut autem voluptas quo   ", // 720
    // "deleniti earum? Qui ipsam ipsum hic ratione mollitia aut nobis laboriosam. Eum aspernatur ", // 810
    // "dolorem sit voluptatum numquam in iure placeat vel laudantium molestiae? Ad reprehenderit ", // 900
    // "quia aut minima deleniti id consequatur sapiente est dolores cupiditate. 012345678901234  ", // 990
  });
  // clang-format on

  auto sv = cudf::strings_column_view(input);
  std::cout << "input size: " << sv.chars_size(cudf::get_default_stream()) << std::endl;

  auto results = nvtext::build_suffix_array(sv);
  auto results_column =
    std::make_unique<cudf::column>(std::move(*(results.release())), rmm::device_buffer{}, 0);
  std::cout << "non-bitonic results: " << results_column->size() << std::endl;
  cudf::test::print(results_column->view());

  results = nvtext::build_suffix_array(sv, true);
  results_column =
    std::make_unique<cudf::column>(std::move(*(results.release())), rmm::device_buffer{}, 0);
  std::cout << "bitonic results: " << results_column->size() << std::endl;
  cudf::test::print(results_column->view());
}
