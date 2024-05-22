/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/memory_resource.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <src/io/utilities/base64_utilities.hpp>

using cudf::io::detail::base64_decode;
using cudf::io::detail::base64_encode;

class IoUtilitiesTest : public cudf::test::BaseFixture {};

TEST(IoUtilitiesTest, HostMemoryGetAndSet)
{
  // Global environment for temporary files
  auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

  // pinned/pooled host memory resource
  using host_pooled_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>;
  host_pooled_mr mr(std::make_shared<rmm::mr::pinned_host_memory_resource>().get(),
                    size_t{128} * 1024 * 1024);

  // set new resource
  auto last_mr = cudf::io::get_host_memory_resource();
  cudf::io::set_host_memory_resource(mr);

  constexpr int num_rows = 32 * 1024;
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 2; });
  auto values = thrust::make_counting_iterator(0);

  cudf::test::fixed_width_column_wrapper<int> col(values, values + num_rows, valids);

  cudf::table_view expected({col});
  auto filepath = temp_env->get_temp_filepath("IoUtilsMemTest.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_args);

  cudf::io::parquet_reader_options const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, expected);

  // reset memory resource back
  cudf::io::set_host_memory_resource(last_mr);
}

TEST(IoUtilitiesTest, Base64EncodeAndDecode)
{
  // a vector of lorem ipsum strings
  std::vector<std::string> strings = {
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut ",
    "labore et dolore magna aliqua. Id ornare arcu odio ut sem. Ultrices neque ornare aenean ",
    "euismod elementum nisi quis. Faucibus pulvinar elementum integer enim. Ut tortor pretium ",
    "viverra suspendisse potenti nullam ac tortor vitae. Elementum pulvinar etiam non quam lacus. ",
    "Fermentum odio eu feugiat pretium nibh. Commodo ullamcorper a lacus vestibulum sed arcu. "
    "Elit ",
    "ut aliquam purus sit amet luctus venenatis lectus magna. Aliquet enim tortor at auctor urna ",
    "nunc id cursus metus. Vivamus at augue eget arcu dictum. Ultricies leo integer malesuada "
    "nunc ",
    "vel risus commodo viverra maecenas.Netus et malesuada fames ac turpis egestas. Erat ",
    "pellentesque adipiscing commodo elit at imperdiet. Commodo nulla facilisi nullam vehicula. ",
    "Morbi tristique senectus et netus et. Cursus vitae congue mauris rhoncus aenean vel elit ",
    "scelerisque mauris. Eros donec ac odio tempor orci dapibus ultrices. Purus in mollis nunc "
    "sed ",
    "id. Justo eget magna fermentum iaculis eu. Diam maecenas ultricies mi eget. Justo laoreet "
    "sit ",
    "amet cursus sit amet. Nibh venenatis cras sed felis eget velit aliquet sagittis id. Dui ut ",
    "ornare lectus sit amet est placerat in egestas. Malesuada nunc vel risus commodo viverra ",
    "maecenas accumsan lacus. Arcu non odio euismod lacinia at. Euismod elementum nisi quis ",
    "eleifend quam adipiscing vitae proin sagittis. Eget sit amet tellus cras adipiscing enim ",
    "eu.Neque ornare aenean euismod elementum nisi quis eleifend quam adipiscing. Posuere ",
    "sollicitudin aliquam ultrices sagittis orci a scelerisque purus. Lobortis elementum nibh ",
    "tellus molestie. Et ligula ullamcorper malesuada proin libero nunc consequat interdum "
    "varius. ",
    "Neque volutpat ac tincidunt vitae semper quis lectus. Nunc mi ipsum faucibus vitae. Congue "
    "eu ",
    "consequat ac felis donec et. Faucibus in ornare quam viverra orci sagittis. Egestas "
    "fringilla ",
    "phasellus faucibus scelerisque eleifend. Sem fringilla ut morbi tincidunt augue. Lobortis ",
    "elementum nibh tellus molestie nunc non. Ultrices neque ornare aenean euismod elementum. ",
    "Cursus turpis massa tincidunt dui ut ornare lectus sit. Eu facilisis sed odio morbi quis "
    "commodo odio. Tortor dignissim convallis aenean et tortor at risus. Sed euismod nisi porta ",
    "lorem. In ornare quam viverra orci sagittis. Sed blandit libero volutpat sed cras. Quis ",
    "viverra nibh cras pulvinar mattis nunc sed blandit libero. Non tellus orci ac auctor augue. ",
    "Mattis molestie a iaculis at erat pellentesque adipiscing. Est lorem ipsum dolor sit amet ",
    "consectetur. Commodo odio aenean sed adipiscing. Nunc lobortis mattis aliquam faucibus "
    "purus. ",
    "Pellentesque massa placerat duis ultricies lacus. Sed viverra tellus in hac habitasse "
    "platea. ",
    "Ut porttitor leo a diam sollicitudin tempor id eu. Rhoncus aenean vel elit scelerisque "
    "mauris ",
    "pellentesque pulvinar pellentesque. Ornare quam viverra orci sagittis. Interdum consectetur ",
    "libero id faucibus nisl tincidunt eget. Eget est lorem ipsum dolor sit amet. Malesuada fames ",
    "ac turpis egestas integer eget aliquet nibh. Scelerisque felis imperdiet proin fermentum "
    "leo. ",
    "Duis convallis convallis tellus id interdum velit. Sit amet massa vitae tortor condimentum ",
    "lacinia quis vel. Eu turpis egestas pretium aenean pharetra. Sed enim ut sem viverra aliquet ",
    "eget sit amet tellus. Feugiat nisl pretium fusce id velit ut tortor. In hendrerit gravida ",
    "rutrum quisque non tellus orci ac auctor. Sit amet nulla facilisi morbi. Nunc congue nisi ",
    "vitae suscipit tellus. Posuere morbi leo urna molestie at elementum eu. Egestas sed tempus ",
    "urna et pharetra pharetra. Sed euismod nisi porta lorem. At elementum eu facilisis sed. Odio ",
    "aenean sed adipiscing diam donec. Congue nisi vitae suscipit tellus mauris a diam. Fringilla ",
    "urna porttitor rhoncus dolor purus non enim praesent. Eget gravida cum sociis natoque. ",
    "Facilisis mauris sit amet massa vitae tortor. Vulputate odio ut enim blandit volutpat ",
    "maecenas volutpat blandit. Ut ornare lectus sit amet est placerat in. Quis vel eros donec ac ",
    "odio tempor orci dapibus ultrices. Venenatis lectus magna fringilla urna porttitor rhoncus ",
    "dolor. Mattis vulputate enim nulla aliquet porttitor lacus. Lectus nulla at volutpat diam ut ",
    "venenatis tellus in. Et ligula ullamcorper malesuada proin libero nunc consequat interdum. "
    "Ut ",
    "enim blandit volutpat maecenas volutpat blandit aliquam etiam erat. Pellentesque pulvinar ",
    "pellentesque habitant morbi tristique senectus et. Auctor eu augue ut lectus arcu bibendum "
    "at ",
    "varius. Posuere ac ut consequat semper viverra nam. Sed euismod nisi porta lorem mollis ",
    "aliquam ut. Porttitor eget dolor morbi non arcu risus quis varius. Adipiscing bibendum est ",
    "ultricies integer quis auctor. Hac habitasse platea dictumst quisque sagittis purus sit amet ",
    "volutpat. Nullam vehicula ipsum a arcu cursus vitae. Velit scelerisque in dictum non ",
    "consectetur a erat nam at. Nulla facilisi cras fermentum odio eu. Tincidunt augue interdum ",
    "velit euismod in pellentesque massa placerat. Suspendisse potenti nullam ac tortor vitae ",
    "purus faucibus ornare. Amet dictum sit amet justo donec enim diam vulputate. Tellus ",
    "pellentesque eu tincidunt tortor aliquam nulla facilisi cras. Mauris in aliquam sem "
    "fringilla ",
    "ut morbi tincidunt. Volutpat diam ut venenatis tellus in metus. Sed pulvinar proin gravida ",
    "hendrerit lectus a. Feugiat nisl pretium fusce id velit ut tortor pretium viverra. Non ",
    "consectetur a erat nam. Fermentum odio eu feugiat pretium nibh ipsum consequat nisl. Donec ",
    "pretium vulputate sapien nec. Purus sit amet luctus venenatis lectus magna fringilla. Mauris ",
    "cursus mattis molestie a iaculis. A iaculis at erat pellentesque adipiscing. Auctor augue ",
    "mauris augue neque gravida in fermentum et sollicitudin. Lectus quam id leo in vitae turpis ",
    "massa sed. Erat nam at lectus urna duis convallis convallis. Dignissim cras tincidunt ",
    "lobortis feugiat vivamus at augue eget arcu. Eleifend mi in nulla posuere sollicitudin ",
    "aliquam ultrices sagittis. Pellentesque nec nam aliquam sem. Feugiat in fermentum posuere ",
    "urna nec tincidunt praesent. Morbi non arcu risus quis varius quam quisque. Morbi tristique ",
    "senectus et netus et malesuada fames ac. Et ligula ullamcorper malesuada proin libero. ",
    "Vivamus at augue eget arcu dictum varius duis at consectetur. Eget mauris pharetra et ",
    "ultrices neque ornare aenean euismod. Sapien faucibus et molestie ac feugiat sed lectus ",
    "vestibulum mattis. Blandit turpis cursus in hac habitasse platea dictumst quisque sagittis. ",
    "Fermentum iaculis eu non diam phasellus vestibulum. Mattis aliquam faucibus purus in massa ",
    "tempor nec feugiat nisl. Lectus sit amet est placerat. Accumsan sit amet nulla facilisi "
    "morbi ",
    "tempus iaculis urna. Magna eget est lorem ipsum dolor sit. Curabitur gravida arcu ac tortor ",
    "dignissim convallis aenean."};

  std::vector<std::string> base64_roundtripped_strings;

  std::transform(strings.begin(),
                 strings.end(),
                 std::back_inserter(base64_roundtripped_strings),
                 [&](auto& str) { return base64_decode(base64_encode(str)); });

  // Create columns for expected and results
  cudf::test::strings_column_wrapper expected(strings.begin(), strings.end());
  cudf::test::strings_column_wrapper results(base64_roundtripped_strings.begin(),
                                             base64_roundtripped_strings.end());

  // Check equal columns
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results);
}
