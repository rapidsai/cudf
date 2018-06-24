
#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <cuda_runtime.h>
#include "test_utils.h"
 

#include <limits.h>

auto chartobin ( unsigned char c ) -> char * {
    static char bin[CHAR_BIT + 1] = {0};
    int i;
    for( i = CHAR_BIT - 1; i >= 0; i-- ) {
        bin[i] = (c % 2) + '0';
        c /= 2;
    }
   return bin;
}

auto print_binary(unsigned char n) -> void {
	std::cout << "decimal: " << (int)n << "\t" << "binary: " << chartobin(n) << std::endl;
}

auto print_column(gdf_column * column) -> void {

    char * host_data_out = new char[column->size];
    gdf_valid_type * host_valid_out;

    if(column->size % 8 != 0){
        host_valid_out = new gdf_valid_type[(column->size + (8 - (column->size % 8)))/8];
    }else{
        host_valid_out = new gdf_valid_type[column->size / 8];
    }

    cudaMemcpy(host_data_out,column->data,sizeof(int8_t) * column->size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_valid_out,column->valid,sizeof(gdf_valid_type) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);

    std::cout<<"Printing Column"<<std::endl;

    int  n_bytes =  sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
    for(int i = 0; i < column->size; i++) {
        int col_position =  i / 8;
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        int bit_offset =  (length_col - 1) - (i % 8);
        std::cout << "host_data_out[" << i << "] = " << ((int)host_data_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
    }
    delete[] host_data_out;
    delete[] host_valid_out;

    std::cout<<std::endl<<std::endl;
}

template<typename ValueType, typename Functor>
auto check_column(gdf_column * column, Functor check_function) -> void  {
    char * host_data_out = new char[column->size]; // TEMPLATE (value_type)
    char * host_valid_out;
    if(column->size % 8 != 0){
        host_valid_out = new char[(column->size + (8 - (column->size % 8)))/8];
    }else{
        host_valid_out = new char[column->size / 8];
    }
    cudaMemcpy(host_data_out,column->data,sizeof(int8_t) * column->size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_valid_out,column->valid,sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);
    int  n_bytes =  sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
    bool* valid = new bool[column->size]; // gdf_valid_type
    for(int i = 0; i < column->size; i++){
        int col_position =  i / 8;
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        int bit_offset =  (length_col - 1) - (i % 8);
        valid[i] = (host_valid_out[col_position] >> bit_offset ) & 1;
    }
    check_function(host_data_out, valid, column->size);
    delete [] valid;
    delete[] host_data_out;
    delete[] host_valid_out;
}
