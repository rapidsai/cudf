#ifndef GDF_GDF_H
#define GDF_GDF_H

#include <cstdlib>
#include <cstdint>
#include "types.h"
#include "convert_types.h"
#include "legacy/io_types.hpp"
#include "legacy/io_functions.hpp"
#include "legacy/io_readers.hpp"
#include "legacy/io_writers.hpp"
#include "legacy/column.hpp"

constexpr size_t GDF_VALID_BITSIZE{(sizeof(gdf_valid_type) * 8)};

extern "C" {
#include "functions.h"
#include "legacy/io_types.h"
#include "legacy/io_functions.h"
}

#endif /* GDF_GDF_H */
