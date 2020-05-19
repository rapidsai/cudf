#ifndef GDF_GDF_H
#define GDF_GDF_H

#include <cstdint>
#include <cstdlib>
#include "convert_types.h"
#include "legacy/column.hpp"
#include "types.h"
#include "types.hpp"

constexpr size_t GDF_VALID_BITSIZE{(sizeof(cudf::valid_type) * 8)};

extern "C" {
#include "legacy/functions.h"
}

#endif /* GDF_GDF_H */
