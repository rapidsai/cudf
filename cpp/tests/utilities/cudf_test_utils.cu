/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cudf_test_utils.cuh"
#include <nvstrings/NVCategory.h>

namespace {

namespace detail {

// When streaming char-like types, the standard library streams tend to treat
// them as characters rather than numbers, e.g. you would get an 'a' instead of 97.
// The following function(s) ensure we "promote" such values to integers before
// they're streamed

template <typename T>
const T& promote_for_streaming(const T& x) { return x; }


//int promote_for_streaming(const char& x)          { return x; }
//int promote_for_streaming(const unsigned char& x) { return x; }
int promote_for_streaming(const signed char& x)   { return x; }

} // namespace detail


struct column_printer {
    template<typename Element>
    void operator()(gdf_column const* the_column, unsigned min_printing_width)
    {
        gdf_size_type num_rows { the_column->size };

        Element const* column_data { static_cast<Element const*>(the_column->data) };

        std::vector<Element> host_side_data(num_rows);
        cudaMemcpy(host_side_data.data(), column_data, num_rows * sizeof(Element), cudaMemcpyDeviceToHost);

        gdf_size_type const num_masks { gdf_valid_allocation_size(num_rows) };
        std::vector<gdf_valid_type> h_mask(num_masks, ~gdf_valid_type { 0 });
        if (nullptr != the_column->valid) {
            cudaMemcpy(h_mask.data(), the_column->valid, num_masks * sizeof(gdf_valid_type), cudaMemcpyDeviceToHost);
        }

        for (gdf_size_type i = 0; i < num_rows; ++i) {
            std::cout << std::setw(min_printing_width);
            if (gdf_is_valid(h_mask.data(), i)) {
                std::cout << detail::promote_for_streaming(host_side_data[i]);
            }
            else {
                std::cout << null_representative;
            }
            std::cout << ' ';
        }
        std::cout << std::endl;

        if(the_column->dtype == GDF_STRING_CATEGORY){
            std::cout<<"Data on category:\n";
            size_t length = 1;

            if(the_column->dtype_info.category != nullptr){
                size_t keys_size = static_cast<NVCategory *>(the_column->dtype_info.category)->keys_size();
                if(keys_size>0){
                    char ** data = new char *[keys_size];
                    for(size_t i=0; i<keys_size; i++){
                        data[i]=new char[length+1];
                    }

                    static_cast<NVCategory *>(the_column->dtype_info.category)->get_keys()->to_host(data, 0, keys_size);

                    for(size_t i=0; i<keys_size; i++){
                        data[i][length]=0;
                    }

                    for(size_t i=0; i<keys_size; i++){
                        std::cout<<"("<<data[i]<<"|"<<i<<")\t";
                    }
                    std::cout<<std::endl;
                }
            }
        }
    }
};
} // namespace

void print_gdf_column(gdf_column const * the_column, unsigned min_printing_width)
{
    cudf::type_dispatcher(the_column->dtype, column_printer{}, the_column, min_printing_width);
}

void print_valid_data(const gdf_valid_type *validity_mask,
                      const size_t num_rows)
{
  cudaError_t error;
  cudaPointerAttributes attrib;
  cudaPointerGetAttributes(&attrib, validity_mask);
  error = cudaGetLastError();

  std::vector<gdf_valid_type> h_mask(gdf_valid_allocation_size(num_rows));
  if (error != cudaErrorInvalidValue && attrib.memoryType == cudaMemoryTypeDevice)
    cudaMemcpy(h_mask.data(), validity_mask, gdf_valid_allocation_size(num_rows), cudaMemcpyDeviceToHost);
  else
    memcpy(h_mask.data(), validity_mask, gdf_valid_allocation_size(num_rows));

  std::transform(
      h_mask.begin(), h_mask.begin() + gdf_num_bitmask_elements(num_rows),
      std::ostream_iterator<std::string>(std::cout, " "), [](gdf_valid_type x) {
        auto bits = std::bitset<GDF_VALID_BITSIZE>(x).to_string('@');
        return std::string(bits.rbegin(), bits.rend());
      });
  std::cout << std::endl;
}

gdf_size_type count_valid_bits_host(
    std::vector<gdf_valid_type> const& masks, gdf_size_type const num_rows)
{
  if ((0 == num_rows) || (0 == masks.size())) {
    return 0;
  }

  gdf_size_type count{0};

  // Count the valid bits for all masks except the last one
  for (gdf_size_type i = 0; i < (gdf_num_bitmask_elements(num_rows) - 1); ++i) {
    gdf_valid_type current_mask = masks[i];

    while (current_mask > 0) {
      current_mask &= (current_mask - 1);
      count++;
    }
  }

  // Only count the bits in the last mask that correspond to rows
  int num_rows_last_mask = num_rows % GDF_VALID_BITSIZE;
  if (num_rows_last_mask == 0) {
    num_rows_last_mask = GDF_VALID_BITSIZE;
  }

  // Mask off only the bits that correspond to rows
  gdf_valid_type const rows_mask = ( gdf_valid_type{1} << num_rows_last_mask ) - 1;
  gdf_valid_type last_mask = masks[gdf_num_bitmask_elements(num_rows) - 1] & rows_mask;

  while (last_mask > 0) {
    last_mask &= (last_mask - 1);
    count++;
  }

  return count;
}
