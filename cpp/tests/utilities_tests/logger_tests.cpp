/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/logger.hpp>

#include <spdlog/sinks/ostream_sink.h>

#include <string>

class LoggerTest : public cudf::test::BaseFixture {
  std::ostringstream oss;
  spdlog::level::level_enum prev_level;
  std::vector<spdlog::sink_ptr> prev_sinks;

 public:
  LoggerTest()
    : prev_level{cudf::detail::logger().level()}, prev_sinks{cudf::detail::logger().sinks()}
  {
    cudf::detail::logger().sinks() = {std::make_shared<spdlog::sinks::ostream_sink_mt>(oss)};
    cudf::detail::logger().set_formatter(
      std::unique_ptr<spdlog::formatter>(new spdlog::pattern_formatter("%v")));
  }
  ~LoggerTest() override
  {
    cudf::detail::logger().set_level(prev_level);
    cudf::detail::logger().sinks() = prev_sinks;
  }

  void clear_sink() { oss.str(""); }
  std::string sink_content() { return oss.str(); }
};

TEST_F(LoggerTest, Basic)
{
  cudf::detail::logger().critical("crit msg");
  ASSERT_EQ(this->sink_content(), "crit msg\n");
}

TEST_F(LoggerTest, DefaultLevel)
{
  cudf::detail::logger().trace("trace");
  cudf::detail::logger().debug("debug");
  cudf::detail::logger().info("info");
  cudf::detail::logger().warn("warn");
  cudf::detail::logger().error("error");
  cudf::detail::logger().critical("critical");
  ASSERT_EQ(this->sink_content(), "warn\nerror\ncritical\n");
}

TEST_F(LoggerTest, CustomLevel)
{
  cudf::detail::logger().set_level(spdlog::level::warn);
  cudf::detail::logger().info("info");
  cudf::detail::logger().warn("warn");
  ASSERT_EQ(this->sink_content(), "warn\n");

  this->clear_sink();

  cudf::detail::logger().set_level(spdlog::level::debug);
  cudf::detail::logger().trace("trace");
  cudf::detail::logger().debug("debug");
  ASSERT_EQ(this->sink_content(), "debug\n");
}
