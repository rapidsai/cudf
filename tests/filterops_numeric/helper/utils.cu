
#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <gtest/gtest.h>
#include "utils.cuh"

ValueType* get_gdf_data_from_device(gdf_column* column) {
    ValueType* host_out = new ValueType[column->size];
    cudaMemcpy(host_out, column->data, sizeof(ValueType) * column->size, cudaMemcpyDeviceToHost);
    return host_out;
}

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
    for (int i = 0; i < n_bytes; i++)
    {
        int length = n_bytes != i + 1 ? GDF_VALID_BITSIZE : column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
        auto result = chartobin(valid[i], length);
        response += std::string(result);
    }
    return response;
}

gdf_valid_type* gen_gdf_valid(size_t column_size)
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
        int i;
        for (i = 0; i < n_bytes - 1; ++i)
        {
            valid[i] = 0b10101111;
        }
        size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
        valid[i] = 1 << length - 1;
    }
    return valid;
}


void delete_gdf_column(gdf_column * column){
    cudaFree(column->data);
    cudaFree(column->valid);
}

gdf_column gen_gdb_column(size_t column_size, ValueType init_value)
{
    char *raw_pointer;
    auto gdf_enum_type_value =  gdf_enum_type_for<ValueType>();
    thrust::device_ptr<ValueType> device_pointer;
    std::tie(raw_pointer, device_pointer) = init_device_vector<char, ValueType>(column_size);

    using thrust::detail::make_normal_iterator;
    thrust::fill(make_normal_iterator(device_pointer), make_normal_iterator(device_pointer + column_size), init_value);

    gdf_valid_type *host_valid = gen_gdf_valid(column_size);
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);

    gdf_valid_type *valid_value_pointer;
    cudaMalloc((void **)&valid_value_pointer, n_bytes);
    cudaMemcpy(valid_value_pointer, host_valid, n_bytes, cudaMemcpyHostToDevice);

    gdf_column output;
    gdf_column_view_augmented(&output, (void *)raw_pointer, valid_value_pointer, column_size, gdf_enum_type_value, 0);
    return output;
}

gdf_column convert_to_device_gdf_column (gdf_column *column) {
    size_t column_size = column->size;
    char *raw_pointer;
    thrust::device_ptr<ValueType> device_pointer;
    std::tie(raw_pointer, device_pointer) = init_device_vector<char, ValueType>(column_size);
 
    void* host_out = column->data;
    cudaMemcpy(raw_pointer, host_out, sizeof(ValueType) * column->size, cudaMemcpyHostToDevice);

    gdf_valid_type *host_valid = column->valid;
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);

    gdf_valid_type *valid_value_pointer;
    cudaMalloc((void **)&valid_value_pointer, n_bytes);
    cudaMemcpy(valid_value_pointer, host_valid, n_bytes, cudaMemcpyHostToDevice);

    gdf_column output;
    gdf_column_view_augmented(&output, (void *)raw_pointer, valid_value_pointer, column_size, column->dtype, column->null_count);
    return output;
}

gdf_column convert_to_host_gdf_column (gdf_column *column) {
    auto host_out = get_gdf_data_from_device(column);
    auto host_valid_out = get_gdf_valid_from_device(column);
    
    auto output = *column;
    output.data = host_out;
    output.valid = host_valid_out;
    return output;
}

void check_column(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
{
    auto lhs_valid = get_gdf_valid_from_device(lhs);
    auto rhs_valid = get_gdf_valid_from_device(rhs);
    auto output_valid = get_gdf_valid_from_device(output);
    
    auto computed = gdf_valid_to_str(output_valid, output->size);
    auto expected = gdf_valid_to_str(lhs_valid, lhs->size) + gdf_valid_to_str(rhs_valid, rhs->size);
    
    delete[] lhs_valid;
    delete[] rhs_valid;
    delete[] output_valid;    
    EXPECT_EQ(computed, expected);
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

auto print_binary(gdf_valid_type n) -> void {
	std::cout << "decimal: " << (int)n << "\t" << "binary: " << chartobin(n) << std::endl;
}

auto print_column(gdf_column * column) -> void {

    auto host_out = get_gdf_data_from_device(column);
    auto host_valid_out = get_gdf_valid_from_device(column);
    std::cout<<"Printing Column"<<std::endl;
    int  n_bytes =  sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
    for(int i = 0; i < column->size; i++) {
        int col_position =  i / 8;
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        int bit_offset =  (length_col - 1) - (i % 8);
        std::cout << "host_out[" << i << "] = " << ((int)host_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
    }
    delete[] host_out;
    delete[] host_valid_out;
    std::cout<<std::endl<<std::endl;
}
/*
template<typename ValueType, typename Functor>
auto check_column(gdf_column * column, Functor check_function) -> void  {
    char * host_out = new char[column->size]; // TEMPLATE (value_type)
    char * host_valid_out;
    if(column->size % 8 != 0){
        host_valid_out = new char[(column->size + (8 - (column->size % 8)))/8];
    }else{
        host_valid_out = new char[column->size / 8];
    }
    cudaMemcpy(host_out,column->data,sizeof(int8_t) * column->size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_valid_out,column->valid,sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);
    int  n_bytes =  sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
    bool* valid = new bool[column->size]; // gdf_valid_type
    for(int i = 0; i < column->size; i++){
        int col_position =  i / 8;
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        int bit_offset =  (length_col - 1) - (i % 8);
        valid[i] = (host_valid_out[col_position] >> bit_offset ) & 1;
    }
    check_function(host_out, valid, column->size);
    delete [] valid;
    delete[] host_out;
    delete[] host_valid_out;
}
*/