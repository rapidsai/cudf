#ifndef NVTX_UTILS_H
#define NVTX_UTILS_H

#include <cstddef> // size_t
#include <cudf/types.h>
#include <cassert>
#include <string>
#include <array>

#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

// TODO: When we switch to a C++ Python interface, switch to using an
// enum class instead of an enum that indexes into an array like this
std::array<const uint32_t, GDF_NUM_COLORS> const colors = {0xff00ff00,0xff0000ff,0xffffff00,0xffff00ff,0xff00ffff,0xffff0000,0xffffffff,0xff006600,0xffffa500};

const gdf_color JOIN_COLOR = GDF_CYAN;
const gdf_color GROUPBY_COLOR = GDF_GREEN;
const gdf_color BINARY_OP_COLOR = GDF_YELLOW;
const gdf_color PARTITION_COLOR = GDF_PURPLE;
const gdf_color READ_CSV_COLOR = GDF_PURPLE;

inline 
void PUSH_RANGE(std::string const & name, const gdf_color color) 
{
#ifdef USE_NVTX
    assert(color < GDF_NUM_COLORS);
    nvtxEventAttributes_t eventAttrib = {0}; 
    eventAttrib.version = NVTX_VERSION; 
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; 
    eventAttrib.colorType = NVTX_COLOR_ARGB; 
    eventAttrib.color = colors[color]; 
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; 
    eventAttrib.message.ascii = name.c_str(); 
    nvtxRangePushEx(&eventAttrib); 
#endif
}

inline 
void PUSH_RANGE(std::string const & name, const uint32_t color) 
{
#ifdef USE_NVTX
    nvtxEventAttributes_t eventAttrib = {0}; 
    eventAttrib.version = NVTX_VERSION; 
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; 
    eventAttrib.colorType = NVTX_COLOR_ARGB; 
    eventAttrib.color = color; 
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; 
    eventAttrib.message.ascii = name.c_str(); 
    nvtxRangePushEx(&eventAttrib); 
#endif
}


inline 
void POP_RANGE(void)
{
#ifdef USE_NVTX
  nvtxRangePop();
#endif
}

#endif
