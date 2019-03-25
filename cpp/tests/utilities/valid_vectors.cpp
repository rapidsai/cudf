/*
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

#include "valid_vectors.h"

host_valid_pointer create_and_init_valid(size_t length, size_t null_count)
{
  auto deleter = [](gdf_valid_type* valid) { delete[] valid; };
  auto n_bytes = gdf_valid_allocation_size(length);
  auto valid_bits = new gdf_valid_type[n_bytes];
   for (size_t i = 0; i < length; ++i) {
    if ((float)std::rand()/(RAND_MAX + 1u) >= (float)null_count/(length-i)) {
      gdf::util::turn_bit_on(valid_bits, i);
    } else {
      gdf::util::turn_bit_off(valid_bits, i);
      --null_count;
    }
  }
  return host_valid_pointer{ valid_bits, deleter };
}

void initialize_valids(std::vector<host_valid_pointer>& valids, size_t size, size_t length, size_t null_count)
{
  valids.clear();
  for (size_t i = 0; i < size; ++i) {
    valids.push_back(create_and_init_valid(length, null_count));
  }
}
