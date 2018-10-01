#ifndef NVTX_UTILS_H
#define NVTX_UTILS_H

#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif

enum class COLOR : uint32_t
{
  GREEN = 0xff00ff00, 
  BLUE = 0xff0000ff,
  YELLOW = 0xffffff00, 
  PURPLE = 0xffff00ff,
  CYAN = 0xff00ffff, 
  RED = 0xffff0000, 
  WHITE = 0xffffffff,
};

constexpr COLOR JOIN_COLOR = COLOR::CYAN;
constexpr COLOR GROUPBY_COLOR = COLOR::GREEN;
constexpr COLOR BINARY_OP_COLOR = COLOR::YELLOW;
constexpr COLOR PARTITION_COLOR = COLOR::PURPLE;

inline 
void PUSH_RANGE(std::string const & name, const COLOR color) 
{
#ifdef USE_NVTX
    nvtxEventAttributes_t eventAttrib = {0}; 
    eventAttrib.version = NVTX_VERSION; 
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; 
    eventAttrib.colorType = NVTX_COLOR_ARGB; 
    eventAttrib.color = static_cast<uint32_t>(color); 
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
