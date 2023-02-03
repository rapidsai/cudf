/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/utilities/logger.hpp>
#include <cudf_test/base_fixture.hpp>

#include <spdlog/sinks/ostream_sink.h>

#include <string>

struct LoggerTest : public cudf::test::BaseFixture {};

namespace {

std::ostringstream sink_ss()
{
  std::ostringstream oss;
  cudf::logger().sinks() = {std::make_shared<spdlog::sinks::ostream_sink_mt>(oss)};
  cudf::logger().set_formatter(
    std::unique_ptr<spdlog::formatter>(new spdlog::pattern_formatter("%v")));
  return oss;
}

}  // namespace

TEST_F(LoggerTest, Basic)
{
  auto ss = sink_ss();
  cudf::logger().critical("crit msg");
  ASSERT_EQ(ss.str(), "crit msg\n");
}

TEST_F(LoggerTest, DefaultLevel)
{
  auto ss = sink_ss();

  cudf::logger().trace("trace");
  cudf::logger().debug("debug");
  cudf::logger().info("info");
  cudf::logger().warn("warn");
  cudf::logger().error("error");
  cudf::logger().critical("critical");
  ASSERT_EQ(ss.str(), "info\nwarn\nerror\ncritical\n");
}

TEST_F(LoggerTest, CustomLevel)
{
  auto ss = sink_ss();

  auto lvl = cudf::logger().level();

  cudf::logger().set_level(spdlog::level::warn);
  cudf::logger().info("info");
  cudf::logger().warn("warn");
  ASSERT_EQ(ss.str(), "warn\n");

  // clear sink
  ss.str(std::string());

  cudf::logger().set_level(spdlog::level::debug);
  cudf::logger().trace("trace");
  cudf::logger().debug("debug");
  ASSERT_EQ(ss.str(), "debug\n");

  // revert to default level
  cudf::logger().set_level(lvl);
}
