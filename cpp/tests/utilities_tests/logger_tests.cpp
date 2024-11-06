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

#include <spdlog/sinks/ostream_sink.h>

#include <string>

#ifdef CUDF_BACKWARDS_COMPATIBILITY
#define GET_LOGGER() cudf::detail::logger()
#define LEVEL_ENUM   spdlog::level
#else
#define GET_LOGGER() cudf::default_logger()
#define LEVEL_ENUM   cudf::level_enum
#endif

class LoggerTest : public cudf::test::BaseFixture {
  std::ostringstream oss;
#ifdef CUDF_BACKWARDS_COMPATIBILITY
  spdlog::level::level_enum prev_level;
  std::vector<spdlog::sink_ptr> prev_sinks;
#else
  cudf::level_enum prev_level;
  spdlog::sink_ptr new_sink;
#endif

 public:
  LoggerTest()
#ifdef CUDF_BACKWARDS_COMPATIBILITY
    : prev_level{GET_LOGGER().level()}, prev_sinks{GET_LOGGER().sinks()}
#else
    : prev_level{GET_LOGGER().level()},
      new_sink{std::make_shared<spdlog::sinks::ostream_sink_mt>(oss)}
#endif
  {
#ifdef CUDF_BACKWARDS_COMPATIBILITY
    GET_LOGGER().sinks() = {std::make_shared<spdlog::sinks::ostream_sink_mt>(oss)};
    GET_LOGGER().set_formatter(
      std::unique_ptr<spdlog::formatter>(new spdlog::pattern_formatter("%v")));
#else
    GET_LOGGER().add_sink(new_sink);
#endif
  }
  ~LoggerTest() override
  {
    GET_LOGGER().set_level(prev_level);
#ifdef CUDF_BACKWARDS_COMPATIBILITY
    GET_LOGGER().sinks() = prev_sinks;
#else
    GET_LOGGER().remove_sink(new_sink);
#endif
  }

  void clear_sink() { oss.str(""); }
  std::string sink_content() { return oss.str(); }
};

TEST_F(LoggerTest, Basic)
{
  GET_LOGGER().critical("crit msg");
  ASSERT_EQ(this->sink_content(), "crit msg\n");
}

TEST_F(LoggerTest, DefaultLevel)
{
  GET_LOGGER().trace("trace");
  GET_LOGGER().debug("debug");
  GET_LOGGER().info("info");
  GET_LOGGER().warn("warn");
  GET_LOGGER().error("error");
  GET_LOGGER().critical("critical");
  ASSERT_EQ(this->sink_content(), "warn\nerror\ncritical\n");
}

TEST_F(LoggerTest, CustomLevel)
{
  GET_LOGGER().set_level(LEVEL_ENUM::warn);
  GET_LOGGER().info("info");
  GET_LOGGER().warn("warn");
  ASSERT_EQ(this->sink_content(), "warn\n");

  this->clear_sink();

  GET_LOGGER().set_level(LEVEL_ENUM::debug);
  GET_LOGGER().trace("trace");
  GET_LOGGER().debug("debug");
  ASSERT_EQ(this->sink_content(), "debug\n");
}
