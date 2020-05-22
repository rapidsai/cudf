/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "jit-cache-test.hpp"

namespace cudf {
namespace test {
TEST_F(JitCacheTest, CacheExceptionTest)
{
  EXPECT_NO_THROW(auto program = getProgram("MemoryCacheTestProg"));
  EXPECT_ANY_THROW(auto program1 = getProgram("MemoryCacheTestProg1"));
}

// Test the in memory caching ability
TEST_F(JitCacheTest, MemoryCacheKernelTest)
{
  // Check the kernel caching

  // Single value column
  // TODO (dm): should be a scalar tho
  auto column = cudf::test::fixed_width_column_wrapper<int>{{5, 0}};
  auto expect = cudf::test::fixed_width_column_wrapper<int>{{125, 0}};

  // make new program and rename it to match old program
  auto program = getProgram("MemoryCacheTestProg1", program2_source);
  // TODO: when I convert this pair to a class, make an inherited test class that can edit names
  std::get<0>(program) = "MemoryCacheTestProg";

  // remove any file cache so below kernel should not be obtained from file
  purgeFileCache();

  // make kernel that if the cache tried to compile, will use a different
  // program than intended and give wrong result.
  auto kernel = getKernelInstantiation("my_kernel", program, {"3", "int"});

  (*std::get<1>(kernel))
    .configure(grid, block)
    .launch(column.operator cudf::mutable_column_view().data<int>());

  cudf::test::expect_columns_equal(expect, column);
}

TEST_F(JitCacheTest, MemoryCacheProgramTest)
{
  // Check program source caching

  // Single value column
  // TODO (dm): should be a scalar tho
  auto column = cudf::test::fixed_width_column_wrapper<int>{{5, 0}};
  auto expect = cudf::test::fixed_width_column_wrapper<int>{{625, 0}};

  // remove any file cache so below program should not be obtained from file
  purgeFileCache();

  auto program = getProgram("MemoryCacheTestProg");
  // make kernel that HAS to be compiled
  auto kernel = getKernelInstantiation("my_kernel", program, {"4", "int"});

  (*std::get<1>(kernel))
    .configure(grid, block)
    .launch(column.operator cudf::mutable_column_view().data<int>());

  cudf::test::expect_columns_equal(expect, column);
}

// Test the file caching ability
#if defined(JITIFY_USE_CACHE)
TEST_F(JitCacheTest, FileCacheProgramTest)
{
  // Brand new cache object that has nothing in in-memory cache
  cudf::jit::cudfJitCache cache;

  // Single value column
  auto column = cudf::test::fixed_width_column_wrapper<int>{{5, 0}};
  auto expect = cudf::test::fixed_width_column_wrapper<int>{{625, 0}};

  // make program
  auto program = cache.getProgram("FileCacheTestProg", program_source);
  // make kernel that HAS to be compiled
  auto kernel = cache.getKernelInstantiation("my_kernel", program, {"4", "int"});
  (*std::get<1>(kernel))
    .configure(grid, block)
    .launch(column.operator cudf::mutable_column_view().data<int>());

  cudf::test::expect_columns_equal(expect, column);
}

TEST_F(JitCacheTest, FileCacheKernelTest)
{
  // Brand new cache object that has nothing in in-memory cache
  cudf::jit::cudfJitCache cache;

  // Single value column
  auto column = cudf::test::fixed_width_column_wrapper<int>{{5, 0}};
  auto expect = cudf::test::fixed_width_column_wrapper<int>{{125, 0}};

  // make program
  auto program = cache.getProgram("FileCacheTestProg", program_source);
  // make kernel that should NOT need to be compiled
  auto kernel = cache.getKernelInstantiation("my_kernel", program, {"3", "int"});
  (*std::get<1>(kernel))
    .configure(grid, block)
    .launch(column.operator cudf::mutable_column_view().data<int>());

  cudf::test::expect_columns_equal(expect, column);
}
#endif

}  // namespace test
}  // namespace cudf

CUDF_TEST_PROGRAM_MAIN()
