
#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <gtest/gtest.h>
#include "utils.cuh"


gdf_valid_type * get_gdf_valid_from_device(gdf_column* column) {
    gdf_valid_type * host_valid_out;
    size_t n_bytes = get_number_of_bytes_for_valid(column->size);
    host_valid_out = new gdf_valid_type[n_bytes];
    cudaMemcpy(host_valid_out,column->valid, n_bytes, cudaMemcpyDeviceToHost);
    return host_valid_out;
}

std::string gdf_valid_to_str(gdf_valid_type *valid, size_t column_size)
{
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    std::string response;
    for (size_t i = 0; i < n_bytes; i++)
    {
        size_t length = (n_bytes != i + 1) ? GDF_VALID_BITSIZE : (column_size - GDF_VALID_BITSIZE * (n_bytes - 1));
        auto result = chartobin(valid[i], length);
        response += std::string(result);
    }
    return response;
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
        size_t n_bytes = get_number_of_bytes_for_valid (column_size);
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
    rmmFree(column->data, 0);
    rmmFree(column->valid, 0);
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

std::string chartobin(gdf_valid_type c, int size/* = 8*/)
{
    std::string bin;
    bin.resize(size);
    bin[0] = 0;
    int i;
    for (i = size - 1; i >= 0; i--)
    {
        bin[i] = (c % 2) + '0';
        c /= 2;
    }
    return bin;
}

auto print_binary(gdf_valid_type n, int size) -> void {
    std::cout << chartobin(n) << "\t sz: " <<  size <<  "\tbinary: " << chartobin(n, size) << std::endl;
}
