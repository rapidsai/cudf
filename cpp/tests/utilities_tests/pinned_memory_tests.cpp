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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

class PinnedMemoryTest : public cudf::test::BaseFixture {
  size_t prev_copy_threshold;
  size_t prev_alloc_threshold;

 public:
  PinnedMemoryTest()
    : prev_copy_threshold{cudf::get_kernel_pinned_copy_threshold()},
      prev_alloc_threshold{cudf::get_allocate_host_as_pinned_threshold()}
  {
  }
  ~PinnedMemoryTest() override
  {
    cudf::set_kernel_pinned_copy_threshold(prev_copy_threshold);
    cudf::set_allocate_host_as_pinned_threshold(prev_alloc_threshold);
  }
};

TEST_F(PinnedMemoryTest, MemoryResourceGetAndSet)
{
  // Global environment for temporary files
  auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

  // pinned/pooled host memory resource
  using host_pooled_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>;
  host_pooled_mr mr(std::make_shared<rmm::mr::pinned_host_memory_resource>().get(),
                    4 * 1024 * 1024);

  // set new resource
  auto last_mr = cudf::get_pinned_memory_resource();
  cudf::set_pinned_memory_resource(mr);

  constexpr int num_rows = 32 * 1024;
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 2; });
  auto values = thrust::make_counting_iterator(0);

  cudf::test::fixed_width_column_wrapper<int> col(values, values + num_rows, valids);

  cudf::table_view expected({col});
  auto filepath = temp_env->get_temp_filepath("MemoryResourceGetAndSetTest.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_args);

  cudf::io::parquet_reader_options const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, expected);

  // reset memory resource back
  cudf::set_pinned_memory_resource(last_mr);
}

TEST_F(PinnedMemoryTest, KernelCopyThresholdGetAndSet)
{
  cudf::set_kernel_pinned_copy_threshold(12345);
  EXPECT_EQ(cudf::get_kernel_pinned_copy_threshold(), 12345);
}

TEST_F(PinnedMemoryTest, HostAsPinnedThresholdGetAndSet)
{
  cudf::set_allocate_host_as_pinned_threshold(12345);
  EXPECT_EQ(cudf::get_allocate_host_as_pinned_threshold(), 12345);
}

TEST_F(PinnedMemoryTest, MakePinnedVector)
{
  cudf::set_allocate_host_as_pinned_threshold(0);

  // should always use pinned memory
  {
    auto const vec = cudf::detail::make_pinned_vector_async<char>(1, cudf::get_default_stream());
    EXPECT_TRUE(vec.get_allocator().is_device_accessible());
  }
}

TEST_F(PinnedMemoryTest, MakeHostVector)
{
  cudf::set_allocate_host_as_pinned_threshold(7);

  // allocate smaller than the threshold
  {
    auto const vec = cudf::detail::make_host_vector<int32_t>(1, cudf::get_default_stream());
    EXPECT_TRUE(vec.get_allocator().is_device_accessible());
  }

  // allocate the same size as the threshold
  {
    auto const vec = cudf::detail::make_host_vector<char>(7, cudf::get_default_stream());
    EXPECT_TRUE(vec.get_allocator().is_device_accessible());
  }

  // allocate larger than the threshold
  {
    auto const vec = cudf::detail::make_host_vector<int32_t>(2, cudf::get_default_stream());
    EXPECT_FALSE(vec.get_allocator().is_device_accessible());
  }
}
