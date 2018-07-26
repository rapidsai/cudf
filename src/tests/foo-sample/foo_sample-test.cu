#include <cstdlib>
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
//#include "../../sort-join.cuh"
#include "../../joining.h"
using namespace std;
using namespace mgpu;
gdf_column
create_gdf_column(mem_t<int> &d) {
      gdf_column c = {d.data(), nullptr, d.size(), GDF_INT32, TIME_UNIT_NONE};
          return c;
}
TEST(gdf_foo_sample_TEST, case1) {
    standard_context_t context;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds=0,sortms=0,hashms = 0;
    for (int countSize=1e3; countSize <=1e8; countSize*=10){ 
        int countA=countSize;
        int countB=countSize;
        for(int maxkey=1e4; maxkey<=1e8; maxkey*=10){
            mem_t<int> dataA = fill_random(0, maxkey, countA, false, context);
            mem_t<int> dataB = fill_random(0, maxkey, countB, false, context);
            cudaEventRecord(start); 
            mergesort(dataA.data(), countA, less_t<int>(), context);
            mergesort(dataB.data(), countB, less_t<int>(), context);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&sortms, start, stop);
            // printf("Sorting, %10d, %10d, %8.5f, %10.0f\n", 
            printf("Sorting,%d,%d,%f,%f\n", 
                countA, maxkey, sortms, float(1000*countA)/(float)sortms);
            mem_t<int> common;
            cudaEventRecord(start);
            //common = inner_join(dataA.data(), countA, dataB.data(), countB, less_t<int>() , context);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            // printf("time: %f  - common elements %d\n", milliseconds,common.size());
            mem_t<int> dataA1 = fill_random(0, maxkey, countA, false, context);
            mem_t<int> dataA2 = fill_random(0, maxkey, countA, false, context);
            mem_t<int> dataA3 = fill_random(0, maxkey, countA, false, context);
            mem_t<int> dataB1 = fill_random(0, maxkey, countB, false, context);
            mem_t<int> dataB2 = fill_random(0, maxkey, countB, false, context);
            mem_t<int> dataB3 = fill_random(0, maxkey, countB, false, context);
            gdf_column gdl0 = create_gdf_column(dataA1);
            gdf_column gdl1 = create_gdf_column(dataA2);
            gdf_column gdl2 = create_gdf_column(dataA3);
            gdf_column gdr0 = create_gdf_column(dataB1);
            gdf_column gdr1 = create_gdf_column(dataB2);
            gdf_column gdr2 = create_gdf_column(dataB3);
            gdf_column* gl[3] = {&gdl0, &gdl1, &gdl2};
            gdf_column* gr[3] = {&gdr0, &gdr1, &gdr2};
            gdf_join_result_type *out;
            // gdf_error err = gdf_multi_left_join_generic(1, gl, gr, &out);
            cudaEventRecord(start); 
            gdf_error err = gdf_multi_left_join_generic(1, gl, gr, &out);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&hashms, start, stop);
            // printf("Hashing, %10d, %10d, %8.5f, %10.0f\n", 
            printf("Hashing,%d,%d,%f,%f\n", 
                countA,  maxkey, hashms, float(1000*countA)/(float)hashms);
                          
            // gdf_error err = gdf_inner_join_i32(&gdl0, &gdr0 , &out);
        }
    }
}
