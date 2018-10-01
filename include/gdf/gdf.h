#ifndef GDF_GDF_H
#define GDF_GDF_H

#include <cstdlib>
#include <cstdint>
#include "cffi/types.h"
#include "cffi/io_types.h"
#include "cffi/convert_types.h"

constexpr size_t GDF_VALID_BITSIZE{(sizeof(gdf_valid_type) * 8)};

extern "C" {
#include "cffi/functions.h"
#include "cffi/io_functions.h"
}

#endif /* GDF_GDF_H */
