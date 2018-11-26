
#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <gtest/gtest.h>
#include "utils.cuh"
#include "../../../util/bit_util.cuh"


gdf_valid_type * get_gdf_valid_from_device(gdf_column* column) {
    if (column->valid == nullptr) {
        return nullptr;
    }

    gdf_valid_type * host_valid_out;
    size_t n_bytes = gdf_get_num_chars_bitmask(column->size);
    host_valid_out = new gdf_valid_type[n_bytes];
    cudaMemcpy(host_valid_out,column->valid, n_bytes, cudaMemcpyDeviceToHost);
    return host_valid_out;
}

gdf_valid_type* gen_gdf_valid(size_t column_size, size_t init_value)
{
    gdf_valid_type *valid = nullptr;
    if (column_size == 0)
    {
        valid = new gdf_valid_type[1];
    }
    else
    {
        size_t n_bytes = gdf_get_num_chars_bitmask (column_size);
        valid = new gdf_valid_type[n_bytes];
        size_t i;
        for (i = 0; i < n_bytes - 1; ++i)
        {
            valid[i] = (init_value % 256);
        }
        size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
        valid[i] = 1 << length - 1;
    }
    return valid;
}


void delete_gdf_column(gdf_column * column){
    RMM_FREE(column->data, 0);
    RMM_FREE(column->valid, 0);
}

gdf_size_type count_zero_bits(gdf_valid_type *valid, size_t column_size)
{
    size_t numbits = 0;
    auto bin = gdf_valid_to_str(valid, column_size);

    for(size_t i = 0; i < bin.length(); i++) {
        if ( bin [i] == '0')
            numbits++;
    }
    return numbits;
}

auto print_binary(gdf_valid_type n, int size) -> void {
    std::cout << chartobin(n) << "\t sz: " <<  size <<  "\tbinary: " << chartobin(n, size) << std::endl;
}

// Create the valid pointer and init randomly the last half column
void initialize_valids(host_valid_pointer& valid_ptr, size_t length, bool all_bits_on)
{
    auto deleter = [](gdf_valid_type* valid) { delete[] valid; };
    auto n_bytes = gdf_get_num_chars_bitmask(length);
    auto valid_bits = new gdf_valid_type[n_bytes];

    for (size_t i = 0; i < length; ++i) {
        if (all_bits_on) {
            gdf::util::turn_bit_on(valid_bits, i);
        } else {
            if (i < length / 2 || std::rand() % 2 == 1) {
               gdf::util::turn_bit_on(valid_bits, i);
            } else {
                gdf::util::turn_bit_off(valid_bits, i);
            }
        }
    }
    valid_ptr = host_valid_pointer{ valid_bits, deleter };
}