
#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <cuda_runtime.h>

#include "utils.cuh"
 


void print_column(gdf_column * column) 
{

    char * host_data_out = new char[column->size];
    char * host_valid_out;

    if(column->size % 8 != 0){
        host_valid_out = new char[(column->size + (8 - (column->size % 8)))/8];
    }else{
        host_valid_out = new char[column->size / 8];
    }


    cudaMemcpy(host_data_out,column->data,sizeof(int8_t) * column->size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_valid_out,column->valid,sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);

    std::cout<<"Printing Column"<<std::endl;

    for(int i = 0; i < column->size; i++){
        int col_position = i / 8;
        int bit_offset = i % 8;
        std::cout<<"host_data_out["<<i<<"] = "<<((int)host_data_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
    }

    delete[] host_data_out;
    delete[] host_valid_out;

    std::cout<<std::endl<<std::endl;
}