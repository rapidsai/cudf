#include <cudf/cudf.h>
#include "nvtx_utils.h"

/* --------------------------------------------------------------------------*/
/**
 * @brief  Start an NVTX range.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * demarcates the begining of a user-defined range with a specified name and
 * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
 * nested within other ranges.
 *
 * @param name The name of the NVTX range
 * @param color The color to use for the range
 *
 * @returns
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_nvtx_range_push(char const * const name, gdf_color color ){

  if((color < 0) || (color > GDF_NUM_COLORS))
    return GDF_UNDEFINED_NVTX_COLOR;

  if(nullptr == name)
    return GDF_NULL_NVTX_NAME;

  PUSH_RANGE(name, color);

  return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Start a NVTX range with a custom ARGB color code.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * demarcates the begining of a user-defined range with a specified name and
 * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
 * nested within other ranges.
 * 
 * @param name The name of the NVTX range
 * @param color The ARGB hex color code to use to color this range (e.g., 0xFF00FF00)
 * 
 * @returns   
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_nvtx_range_push_hex(char const * const name, unsigned int color ){

  if(nullptr == name)
    return GDF_NULL_NVTX_NAME;

  PUSH_RANGE(name, color);

  return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief Ends the inner-most NVTX range.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * will demarcate the end of the inner-most range, i.e., the most recent call to
 * gdf_nvtx_range_push.
 * 
 * @returns   
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_nvtx_range_pop(){
  POP_RANGE();
  return GDF_SUCCESS;
}
