/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "../environment.hpp"

#include <cudf_test/cudf_gtest.hpp>

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/progress_thread.hpp>

#include <memory>

Environment* GlobalEnvironment = nullptr;

Environment::Environment(int, char**) {}

TestEnvironmentType Environment::type() const { return TestEnvironmentType::SINGLE; }

void Environment::SetUp()
{
  options_ = rapidsmpf::config::Options(rapidsmpf::config::get_environment_variables());
  comm_ =
    std::make_shared<rapidsmpf::Single>(options_, std::make_shared<rapidsmpf::ProgressThread>());
  split_comm_ = comm_;
}

void Environment::TearDown()
{
  split_comm_ = nullptr;
  comm_       = nullptr;
}

void Environment::barrier() {}

std::shared_ptr<rapidsmpf::Communicator> Environment::split_comm() { return split_comm_; }

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  GlobalEnvironment = new Environment(argc, argv);
  ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
  return RUN_ALL_TESTS();
}
