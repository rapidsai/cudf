#pragma once

#include "utilities/cudf_utils.h"
#include "utilities/wrapper_types.hpp"
#include "utilities/bit_util.cuh"
#include "bitmask/legacy/legacy_bitmask.hpp"


struct GdfValidToBool {
  gdf_valid_type *d_valid;

  __device__ bool operator()(gdf_size_type idx) {
    return gdf_is_valid(d_valid, idx);
  }
};

struct GdfBoolToValid {
  gdf_valid_type *d_valid;
  bool *d_bools;

  __device__ void operator()(gdf_size_type idx) {
    if (d_bools[idx])
      cudf::util::turn_bit_on(d_valid, idx);
    else
      cudf::util::turn_bit_off(d_valid, idx);
  }
};


rmm::device_vector<bool> get_bools_from_gdf_valid(gdf_column *column);

void set_bools_for_gdf_valid(gdf_column *column,
                             rmm::device_vector<bool> &d_bools);