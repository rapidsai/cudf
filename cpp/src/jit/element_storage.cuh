
#pragma once

namespace cudf {

// can store any nullable or non-nullable element type
struct element_storage {
  alignas(32) unsigned char data[64];
};

}  // namespace cudf
