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

#include <cudf/logger.hpp>

#include <string>

class LoggerTest : public cudf::test::BaseFixture {
  std::ostringstream oss;
  cudf::level_enum prev_level;

 public:
  LoggerTest() : prev_level{cudf::default_logger().level()}
  {
    cudf::default_logger().sinks().push_back(std::make_shared<cudf::ostream_sink_mt>(oss));
    cudf::default_logger().set_pattern("%v");
  }
  ~LoggerTest() override
  {
    cudf::default_logger().set_pattern("[%6t][%H:%M:%S:%f][%-6l] %v");
    cudf::default_logger().set_level(prev_level);
    cudf::default_logger().sinks().pop_back();
  }

  void clear_sink() { oss.str(""); }
  std::string sink_content() { return oss.str(); }
};

TEST_F(LoggerTest, Basic)
{
  cudf::default_logger().critical("crit msg");
  ASSERT_EQ(this->sink_content(), "crit msg\n");
}

TEST_F(LoggerTest, DefaultLevel)
{
  cudf::default_logger().trace("trace");
  cudf::default_logger().debug("debug");
  cudf::default_logger().info("info");
  cudf::default_logger().warn("warn");
  cudf::default_logger().error("error");
  cudf::default_logger().critical("critical");
  ASSERT_EQ(this->sink_content(), "info\nwarn\nerror\ncritical\n");
}

TEST_F(LoggerTest, CustomLevel)
{
  cudf::default_logger().set_level(cudf::level_enum::warn);
  cudf::default_logger().info("info");
  cudf::default_logger().warn("warn");
  ASSERT_EQ(this->sink_content(), "warn\n");

  this->clear_sink();

  cudf::default_logger().set_level(cudf::level_enum::debug);
  cudf::default_logger().trace("trace");
  cudf::default_logger().debug("debug");
  ASSERT_EQ(this->sink_content(), "debug\n");
}
