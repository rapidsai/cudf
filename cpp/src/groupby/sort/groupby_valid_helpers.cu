#include "groupby_valid_helpers.h"

#include "utilities/cudf_utils.h"
#include "utilities/wrapper_types.hpp"
#include "utilities/bit_util.cuh"
#include "bitmask/legacy/legacy_bitmask.hpp"
 

rmm::device_vector<bool> get_bools_from_gdf_valid(gdf_column *column) {
  rmm::device_vector<bool> d_bools(column->size);
  thrust::transform(
      thrust::make_counting_iterator(static_cast<gdf_size_type>(0)),
      thrust::make_counting_iterator(column->size), d_bools.begin(),
      GdfValidToBool{column->valid});
  return d_bools;
}

void set_bools_for_gdf_valid(gdf_column *column,
                             rmm::device_vector<bool> &d_bools) {
  thrust::for_each(
      thrust::make_counting_iterator(static_cast<gdf_size_type>(0)),
      thrust::make_counting_iterator(column->size),
      GdfBoolToValid{column->valid, d_bools.data().get()});
}