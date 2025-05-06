#pragma once

#include <cudf/io/types.hpp>
#include <cudf/utilities/error.hpp>

#include <gtest/gtest.h>

#include <cstdlib>  // for setenv/unsetenv
#include <string>
#include <tuple>

template <typename Base>
struct CompressionTest
  : public Base,
    public ::testing::WithParamInterface<std::tuple<std::string, cudf::io::compression_type>> {
  CompressionTest()
  {
    auto const comp_impl = std::get<0>(GetParam());

    if (comp_impl == "NVCOMP") {
      setenv("LIBCUDF_HOST_COMPRESSION", "OFF", 1);
      setenv("LIBCUDF_NVCOMP_POLICY", "ALWAYS", 1);
    } else if (comp_impl == "DEVICE_INTERNAL") {
      setenv("LIBCUDF_HOST_COMPRESSION", "OFF", 1);
      setenv("LIBCUDF_NVCOMP_POLICY", "OFF", 1);
    } else if (comp_impl == "HOST") {
      setenv("LIBCUDF_HOST_COMPRESSION", "ON", 1);
      setenv("LIBCUDF_NVCOMP_POLICY", "OFF", 1);
    } else {
      CUDF_FAIL("Invalid test parameter");
    }
  }
  ~CompressionTest() override
  {
    unsetenv("LIBCUDF_HOST_COMPRESSION");
    unsetenv("LIBCUDF_NVCOMP_POLICY");
  }
};