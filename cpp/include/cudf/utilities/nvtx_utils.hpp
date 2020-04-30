#pragma once

#include <cstdint>

namespace cudf {
namespace nvtx {
enum class color : uint32_t {
  GREEN      = 0xff00ff00,
  BLUE       = 0xff0000ff,
  YELLOW     = 0xffffff00,
  PURPLE     = 0xffff00ff,
  CYAN       = 0xff00ffff,
  RED        = 0xffff0000,
  WHITE      = 0xffffffff,
  DARK_GREEN = 0xff006600,
  ORANGE     = 0xffffa500,
};

constexpr color JOIN_COLOR      = color::CYAN;
constexpr color GROUPBY_COLOR   = color::GREEN;
constexpr color BINARY_OP_COLOR = color::YELLOW;
constexpr color PARTITION_COLOR = color::PURPLE;
constexpr color READ_CSV_COLOR  = color::PURPLE;

/**
 * @brief  Start an NVTX range.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * demarcates the begining of a user-defined range with a specified name and
 * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
 * nested within other ranges.
 *
 * @throws cudf::logic_error if `name` is null
 *
 * @param[in] name The name of the NVTX range
 * @param[in] color The color to use for the range
 **/
void range_push(const char* name, color color);

/**
 * @brief  Start a NVTX range with a custom ARGB color code.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * demarcates the begining of a user-defined range with a specified name and
 * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
 * nested within other ranges.
 *
 * @throws cudf::logic_error if `name` is null
 *
 * @param[in] name The name of the NVTX range
 * @param[in] color The ARGB hex color code to use to color this range (e.g., 0xFF00FF00)
 **/
void range_push_hex(const char* name, uint32_t color);

/**
 * @brief Ends the inner-most NVTX range.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * will demarcate the end of the inner-most range, i.e., the most recent call to
 * range_push.
 **/
void range_pop();

}  // namespace nvtx
}  // namespace cudf
