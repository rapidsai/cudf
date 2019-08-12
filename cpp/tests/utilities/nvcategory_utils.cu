/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Felipe Aramburu <felipe@blazingdb.com>
 *     Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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

#include <cudf/cudf.h>
#include <cudf/functions.h>
#include <cudf/types.h>
#include <bitmask/legacy/bit_mask.cuh>
#include <utilities/cudf_utils.h>
#include <utilities/column_utils.hpp>

#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/nvcategory_utils.cuh>

#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>

#include <rmm/rmm.h>

#include <iostream>
#include <random>
#include <cstring>

namespace cudf {
namespace test {

std::string random_string(size_t len, std::string const &allowed_chars) {
  std::mt19937_64 gen { std::random_device()() };
  std::uniform_int_distribution<size_t> dist { 0, allowed_chars.length()-1 };

  std::string ret;
  std::generate_n(std::back_inserter(ret), len, [&] { return allowed_chars[dist(gen)]; });
  return ret;
}

gdf_column * create_nv_category_column(gdf_size_type num_rows, bool repeat_strings){

  const char ** string_host_data = new const char *[num_rows];


  for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
    string_host_data[row_index] = new char[(num_rows + 25) / 26]; //allows string to grow depending on numbe of rows
    std::string temp_string = "";
    int num_chars = repeat_strings ? 1 : (row_index / 26) + 1;
    char repeat_char = (26 - (row_index % 26)) + 65; //chars are Z,Y ...C,B,A,ZZ,YY,.....BBB,AAA.....
    for(int char_index = 0; char_index < num_chars; char_index++){
      temp_string.push_back(repeat_char);
    }
    temp_string.push_back(0);
    std::memcpy((void *) string_host_data[row_index],temp_string.c_str(),temp_string.size());

  }

  NVCategory* category = NVCategory::create_from_array(string_host_data, num_rows);

  gdf_column * column = new gdf_column{};
  int * data;
  RMM_ALLOC(&data, num_rows * sizeof(gdf_nvstring_category) , 0);


  category->get_values( (int *)data, true );
  bit_mask::bit_mask_t * valid;
  bit_mask::create_bit_mask(&valid, num_rows,1);

  gdf_error err = gdf_column_view(column,
      (void *) data,
      (gdf_valid_type *)valid,
      num_rows,
      GDF_STRING_CATEGORY);
  column->dtype_info.category = category;
  return column;
}

gdf_column * create_nv_category_column_strings(const char ** string_host_data, gdf_size_type num_rows){
  NVCategory* category = NVCategory::create_from_array(string_host_data, num_rows);

  gdf_column * column = new gdf_column{};
  int * data;
  RMM_ALLOC(&data, num_rows * sizeof(gdf_nvstring_category) , 0);

  category->get_values( (int *)data, true );
  bit_mask::bit_mask_t * valid;
  bit_mask::create_bit_mask(&valid, num_rows,1);

  gdf_error err = gdf_column_view(column,
      (void *) data,
      (gdf_valid_type *)valid,
      num_rows,
      GDF_STRING_CATEGORY);
  column->dtype_info.category = category;
  return column;
}

const char ** generate_string_data(gdf_size_type num_rows, size_t length, bool print){
  const char ** string_host_data = new const char *[num_rows];

  for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
    string_host_data[row_index] = new char[length+1];

    std::string rand_string = cudf::test::random_string(length);
    rand_string.push_back(0);
    if(print)
      std::cout<<rand_string<<"\t";
    std::memcpy((void *) string_host_data[row_index],rand_string.c_str(),rand_string.size());
  }
  if(print)
    std::cout<<std::endl;

  return string_host_data;
}

std::tuple<std::vector<std::string>, std::vector<gdf_valid_type>> nvcategory_column_to_host(gdf_column * column){

  if (column->dtype == GDF_STRING_CATEGORY && column->dtype_info.category != nullptr && column->size > 0) {
    NVStrings* tptr = static_cast<NVCategory*>(column->dtype_info.category)->gather_strings(static_cast<nv_category_index_type*>(column->data),
                                                                                            column->size,
                                                                                            DEVICE_ALLOCATED);

    unsigned int count = tptr->size();
    if( count==0 )
        return std::make_tuple(std::vector<std::string>(), std::vector<gdf_valid_type>());

    std::vector<char*> list(count);
    char** plist = list.data();
    std::vector<int> lens(count);
    size_t totalmem = tptr->byte_count(lens.data(),false);
    std::vector<char> buffer(totalmem+count,0); // null terminates each string
    char* pbuffer = buffer.data();
    size_t offset = 0;
    for( unsigned int idx=0; idx < count; ++idx )
    {
        plist[idx] = pbuffer + offset;
        offset += lens[idx]+1; // account for null-terminator; also nulls are -1
    }
    tptr->to_host(plist,0,count);

    // TODO: workaround for custrings issue #330. Remove once fix is merged
    // workaround just resets the nullptr entries back to their proper offsets 
    // so that the std::vector constructor below can succeed.
    offset = 0;
    for( unsigned int idx=0; idx < count; ++idx )
    {
        plist[idx] = pbuffer + offset;
        offset += lens[idx]+1; // account for null-terminator; also nulls are -1
    }

    NVStrings::destroy(tptr);
    std::vector<std::string> host_strings_vector(plist, plist + column->size);
   
    std::vector<gdf_valid_type> host_bitmask(gdf_valid_allocation_size(column->size));
    if (cudf::is_nullable(*column)) {
      CUDA_TRY(cudaMemcpy(host_bitmask.data(),
                          column->valid,
                          host_bitmask.size()*sizeof(gdf_valid_type),
                          cudaMemcpyDeviceToHost));
    }
    return std::make_tuple(host_strings_vector, host_bitmask);
  } else {
    return std::make_tuple(std::vector<std::string>(), std::vector<gdf_valid_type>());
  }
}


} // namespace test
} // namespace cudf
