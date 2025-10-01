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

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/utilities/error.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <list>
#include <string>
#include <tuple>

class tmp_env_var {
 public:
  explicit tmp_env_var(std::string name, std::string const& value) : name_(std::move(name))
  {
    auto const previous_value = std::getenv(name_.c_str());
    if (previous_value != nullptr) { previous_value_ = std::string(previous_value); }

    setenv(name_.c_str(), value.c_str(), 1);
  }

  tmp_env_var(tmp_env_var const&)            = delete;
  tmp_env_var& operator=(tmp_env_var const&) = delete;
  tmp_env_var(tmp_env_var&&)                 = delete;
  tmp_env_var& operator=(tmp_env_var&&)      = delete;

  ~tmp_env_var()
  {
    if (previous_value_.has_value()) {
      setenv(name_.c_str(), previous_value_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previous_value_;
};

static constexpr char const* host_comp_env_var     = "LIBCUDF_HOST_COMPRESSION";
static constexpr char const* host_decomp_env_var   = "LIBCUDF_HOST_DECOMPRESSION";
static constexpr char const* nvcomp_policy_env_var = "LIBCUDF_NVCOMP_POLICY";

template <typename Base>
struct CompressionTest
  : public Base,
    public ::testing::WithParamInterface<std::tuple<std::string, cudf::io::compression_type>> {
  CompressionTest()
  {
    auto const comp_impl = std::get<0>(GetParam());

    if (comp_impl == "NVCOMP") {
      env_vars.emplace_back(host_comp_env_var, "OFF");
      env_vars.emplace_back(nvcomp_policy_env_var, "ALWAYS");
    } else if (comp_impl == "DEVICE_INTERNAL") {
      env_vars.emplace_back(host_comp_env_var, "OFF");
      env_vars.emplace_back(nvcomp_policy_env_var, "OFF");
    } else if (comp_impl == "HOST") {
      env_vars.emplace_back(host_comp_env_var, "ON");
    } else if (comp_impl == "HYBRID") {
      env_vars.emplace_back(host_comp_env_var, "HYBRID");
    } else if (comp_impl == "AUTO") {
      env_vars.emplace_back(host_comp_env_var, "AUTO");
    } else {
      CUDF_FAIL("Invalid test parameter");
    }
  }

 private:
  std::list<tmp_env_var> env_vars;
};

template <typename Base>
struct DecompressionTest
  : public Base,
    public ::testing::WithParamInterface<std::tuple<std::string, cudf::io::compression_type>> {
  DecompressionTest()
  {
    auto const comp_impl = std::get<0>(GetParam());

    if (comp_impl == "NVCOMP") {
      env_vars.emplace_back(host_decomp_env_var, "OFF");
      env_vars.emplace_back(nvcomp_policy_env_var, "ALWAYS");
    } else if (comp_impl == "DEVICE_INTERNAL") {
      env_vars.emplace_back(host_decomp_env_var, "OFF");
      env_vars.emplace_back(nvcomp_policy_env_var, "OFF");
    } else if (comp_impl == "HOST") {
      env_vars.emplace_back(host_decomp_env_var, "ON");
    } else if (comp_impl == "HYBRID") {
      env_vars.emplace_back(host_decomp_env_var, "HYBRID");
    } else if (comp_impl == "AUTO") {
      env_vars.emplace_back(host_decomp_env_var, "AUTO");
    } else {
      CUDF_FAIL("Invalid test parameter");
    }
  }

 private:
  std::list<tmp_env_var> env_vars;
};
