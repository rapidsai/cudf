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

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace {

class mr_adapter : public rmm::mr::device_memory_resource {
 public:
  mr_adapter(rmm::mr::device_memory_resource* resource_to_wrap) : resource(resource_to_wrap) {}

 private:
  rmm::mr::device_memory_resource* const resource;

  void throw_exception([[maybe_unused]] std::string s)
  {
    throw std::runtime_error("Intentionally throwing exception: " + s);
  }

 protected:
  void* do_allocate(std::size_t num_bytes, rmm::cuda_stream_view stream) override
  {
    void* result;

    try {
      result = resource->allocate(num_bytes, stream);
    } catch (rmm::out_of_memory const& e) {
      printf("OOM caught\n");
      fflush(stdout);
      throw;
    }

    // Actually we need not to throw here, so we will have the newly allocated memory not to be
    // deallocated by `resource`.
    try {
      // throw_exception("allocation");
    } catch (std::exception const& e) {
      printf("Exception caught: %s. Deallocating...\n", e.what());
      fflush(stdout);

      resource->deallocate(result, num_bytes, stream);

      printf("Deallocated\n");
      fflush(stdout);

      throw;
    }

    return result;
  }

  void do_deallocate(void* p, std::size_t size, rmm::cuda_stream_view stream) override
  {
    resource->deallocate(p, size, stream);

    // Throw exception out of this func will lead to `interrupted by signal 11:SIGSEGV`
    throw_exception("deallocation");
  }
};

}  // namespace

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

struct Tests : public cudf::test::BaseFixture {};

TEST_F(Tests, Test)
{
  {
    auto cuda_mr = new rmm::mr::cuda_memory_resource();
    auto pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
      cuda_mr, 1024 * 1024L, 1024 * 1024L);
    auto mr = new mr_adapter(pool_mr);
    cudf::set_current_device_resource(mr);
  }
  try {
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    auto c_stream                     = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(0));
    [[maybe_unused]] void* ret =
      mr.allocate_async(6 * 1024L, rmm::CUDA_ALLOCATION_ALIGNMENT, c_stream);

    // This triggers SIGSEGV.
    mr.deallocate_async(ret, 6 * 1024L, c_stream);
  } catch (...) {  // suppress all exceptions
  }
}
