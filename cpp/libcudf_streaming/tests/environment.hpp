/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf_test/cudf_gtest.hpp>

#include <rapidsmpf/communicator/communicator.hpp>

enum class TestEnvironmentType : int {
  SINGLE,
};

class Environment : public ::testing::Environment {
 public:
  Environment(int argc, char** argv);

  void SetUp() override;

  void TearDown() override;

  void barrier();

  [[nodiscard]] TestEnvironmentType type() const;

  constexpr rapidsmpf::config::Options& options() { return options_; }

  std::shared_ptr<rapidsmpf::Communicator> split_comm();

  std::shared_ptr<rapidsmpf::Communicator> comm_;

 private:
  std::shared_ptr<rapidsmpf::Communicator> split_comm_{nullptr};
  rapidsmpf::config::Options options_;
};

extern Environment* GlobalEnvironment;
