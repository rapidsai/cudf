/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <type_traits>
#include <utility>

namespace {

class scoped_current_device_resource {
 public:
  template <typename Resource>
  explicit scoped_current_device_resource(Resource& resource)
    : _previous{cudf::set_current_device_resource(resource)}
  {
  }

  ~scoped_current_device_resource()
  {
    auto replaced = cudf::set_current_device_resource(std::move(_previous));
    static_cast<void>(replaced);
  }

  scoped_current_device_resource(scoped_current_device_resource const&)            = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource const&) = delete;

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> _previous;
};

rmm::device_async_resource_ref get_output_mr(cudf::memory_resources resources)
{
  return resources.get_output_mr();
}

static_assert(std::is_nothrow_copy_constructible_v<cudf::memory_resources>);
static_assert(std::is_nothrow_move_constructible_v<cudf::memory_resources>);
static_assert(noexcept(std::declval<cudf::memory_resources const&>().get_output_mr()));
static_assert(noexcept(std::declval<cudf::memory_resources const&>().get_temporary_mr()));

TEST(MemoryResourcesTest, OneResourceUsesCurrentResourceAtConstruction)
{
  auto upstream       = cudf::get_current_device_resource_ref();
  auto output_mr      = rmm::mr::statistics_resource_adaptor{upstream};
  auto temporary_mr   = rmm::mr::statistics_resource_adaptor{upstream};
  auto replacement_mr = rmm::mr::statistics_resource_adaptor{upstream};

  scoped_current_device_resource temporary_scope{temporary_mr};
  cudf::memory_resources resources{output_mr};

  EXPECT_TRUE(resources.get_output_mr() == rmm::device_async_resource_ref{output_mr});
  EXPECT_TRUE(resources.get_temporary_mr() == rmm::device_async_resource_ref{temporary_mr});

  {
    scoped_current_device_resource replacement_scope{replacement_mr};
    cudf::memory_resources replacement_resources{output_mr};
    EXPECT_TRUE(replacement_resources.get_temporary_mr() ==
                rmm::device_async_resource_ref{replacement_mr});
  }
}

TEST(MemoryResourcesTest, TwoResourcesIgnoreCurrentResource)
{
  auto upstream     = cudf::get_current_device_resource_ref();
  auto output_mr    = rmm::mr::statistics_resource_adaptor{upstream};
  auto temporary_mr = rmm::mr::statistics_resource_adaptor{upstream};
  auto unrelated_mr = rmm::mr::statistics_resource_adaptor{upstream};

  scoped_current_device_resource current_scope{unrelated_mr};
  cudf::memory_resources resources{output_mr, temporary_mr};

  EXPECT_TRUE(resources.get_output_mr() == rmm::device_async_resource_ref{output_mr});
  EXPECT_TRUE(resources.get_temporary_mr() == rmm::device_async_resource_ref{temporary_mr});
  EXPECT_FALSE(resources.get_temporary_mr() == cudf::get_current_device_resource_ref());
}

TEST(MemoryResourcesTest, ResourceObjectAndRefCompatibility)
{
  auto output_mr    = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
  auto temporary_mr = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
  auto output_ref   = rmm::device_async_resource_ref{output_mr};
  auto temporary_ref = rmm::device_async_resource_ref{temporary_mr};

  static_assert(std::is_convertible_v<decltype(output_mr)&, cudf::memory_resources>);
  static_assert(std::is_convertible_v<rmm::device_async_resource_ref, cudf::memory_resources>);
  static_assert(
    std::is_constructible_v<cudf::memory_resources, decltype(output_mr)&, decltype(temporary_mr)&>);

  cudf::memory_resources from_objects{output_mr, temporary_mr};
  cudf::memory_resources from_refs{output_ref, temporary_ref};
  auto copied = from_objects;

  EXPECT_TRUE(from_refs.get_output_mr() == output_ref);
  EXPECT_TRUE(from_refs.get_temporary_mr() == temporary_ref);
  EXPECT_TRUE(copied.get_output_mr() == output_ref);
  EXPECT_TRUE(copied.get_temporary_mr() == temporary_ref);
  EXPECT_TRUE(get_output_mr(output_mr) == output_ref);
  EXPECT_TRUE(get_output_mr(output_ref) == output_ref);
}

}  // namespace
