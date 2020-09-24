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

#pragma once

#include <boost/filesystem.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <jit/cache.h>

// Note that this test does not inherit from cudf::test::BaseFixture because
// doing so would cause the CUDA context to be created before the fork in
// the JitCacheMultiProcessTest, where we need it to be created after the fork
// to ensure the forked child has a context. These tests do not need the
// memory_resource member of BaseFixture.
struct JitCacheTest : public ::testing::Test, public cudf::jit::cudfJitCache {
  JitCacheTest() : grid(1), block(1) {}

  virtual ~JitCacheTest() {}

  virtual void SetUp()
  {
    purgeFileCache();
    warmUp();
  }

  virtual void TearDown() { purgeFileCache(); }

  void purgeFileCache()
  {
#if defined(JITIFY_USE_CACHE)
    boost::filesystem::remove_all(cudf::jit::getCacheDir());
#endif
  }

  void warmUp()
  {
    // Prime up the cache so that the in-memory and file cache is populated

    // Single value column
    auto column = cudf::test::fixed_width_column_wrapper<int>({4, 0});
    auto expect = cudf::test::fixed_width_column_wrapper<int>({64, 0});

    // make program
    auto program = getProgram("MemoryCacheTestProg", program_source);
    // make kernel
    auto kernel = getKernelInstantiation("my_kernel", program, {"3", "int"});
    (*std::get<1>(kernel))
      .configure(grid, block)
      .launch(column.operator cudf::mutable_column_view().data<int>());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, column);
  }

  const char* program_source =
    "my_program\n"
    "template<int N, typename T>\n"
    "__global__\n"
    "void my_kernel(T* data) {\n"
    "    T data0 = data[0];\n"
    "    for( int i=0; i<N-1; ++i ) {\n"
    "        data[0] *= data0;\n"
    "    }\n"
    "}\n";

  const char* program2_source =
    "my_program\n"
    "template<int N, typename T>\n"
    "__global__\n"
    "void my_kernel(T* data) {\n"
    "    T data0 = data[0];\n"
    "    for( int i=0; i<N-1; ++i ) {\n"
    "        data[0] += data0;\n"
    "    }\n"
    "}\n";

  const char* program3_source =
    "my_program\n"
    "template<int N, typename T>\n"
    "__global__\n"
    "void my_kernel(T* data, T* out) {\n"
    "    T data0 = data[0];\n"
    "    for( int i=0; i<N; ++i ) {\n"
    "        out[0] *= data0;\n"
    "    }\n"
    "}\n";

  dim3 grid;
  dim3 block;
};

/**
 * @brief Similar to JitCacheTest but it doesn't run warmUp() test in SetUp and
 * purgeFileCache() in TearDown
 **/
struct JitCacheMultiProcessTest : public JitCacheTest {
  virtual void SetUp() { purgeFileCache(); }

  virtual void TearDown() {}
};
