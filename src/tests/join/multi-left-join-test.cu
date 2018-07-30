#include <cstdlib>
#include <iostream>
#include <vector>
#include <functional>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include "../../joining.h"


using namespace testing;
using namespace std;
using namespace mgpu;


template <typename T>
struct non_negative
{
  __host__ __device__
  bool operator()(const T x)
  {
    return (x >= 0);
  }
};

template <typename T>
struct EnumType          { static const gdf_dtype type { N_GDF_TYPES }; };
template <> struct EnumType<int8_t>  { static const gdf_dtype type { GDF_INT8    }; };
template <> struct EnumType<int16_t> { static const gdf_dtype type { GDF_INT16   }; };
template <> struct EnumType<int32_t> { static const gdf_dtype type { GDF_INT32   }; };
template <> struct EnumType<int64_t> { static const gdf_dtype type { GDF_INT64   }; };
template <> struct EnumType<float>   { static const gdf_dtype type { GDF_FLOAT32 }; };
template <> struct EnumType<double>  { static const gdf_dtype type { GDF_FLOAT64 }; };

template <typename T>
gdf_column
create_gdf_column(thrust::device_vector<T> &d) {
    gdf_column c = {thrust::raw_pointer_cast(d.data()), nullptr, d.size(), EnumType<T>::type, TIME_UNIT_NONE};
    return c;
}

template <typename T>
gdf_column
create_gdf_column(mem_t<T> &d) {
      gdf_column c = {d.data(), nullptr, d.size(), EnumType<T>::type, TIME_UNIT_NONE};
          return c;
}

template <typename T>
std::vector<T> host_vec(thrust::device_vector<T> &dev_vec) {
    std::vector<T> data(dev_vec.size());
    thrust::copy(dev_vec.begin(), dev_vec.end(), data.begin());
    return data;
}

template <typename T>
gdf_error
call_gdf_single_column_test(
        const std::vector<T> &l,
        const std::vector<T> &r,
        thrust::device_vector<int> &out_left_pos,
        thrust::device_vector<int> &out_right_pos,
        thrust::device_vector<T> &l_idx,
        thrust::device_vector<T> &r_idx,
        const std::function<gdf_error(gdf_column *, gdf_column *,
            gdf_join_result_type **)> &f) {
    thrust::device_vector<T> dl = l;
    thrust::device_vector<T> dr = r;

    gdf_column gdl = create_gdf_column(dl);
    gdf_column gdr = create_gdf_column(dr);

    gdf_join_result_type *out;
    gdf_error err = f(&gdl, &gdr, &out);
    size_t len = gdf_join_result_size(out);
    size_t hlen = len/2;
    int* out_ptr = reinterpret_cast<int*>(gdf_join_result_data(out));
    thrust::device_vector<int> out_data(out_ptr, out_ptr + len);

    thrust::sort_by_key(out_data.begin() + hlen, out_data.end(), out_data.begin());
    thrust::sort_by_key(out_data.begin(), out_data.begin() + hlen, out_data.begin() + hlen);
    out_left_pos.resize(hlen);
    out_right_pos.resize(hlen);
    thrust::copy(out_data.begin(), out_data.begin() + out_left_pos.size(), out_left_pos.begin());
    thrust::copy(out_data.begin() + out_right_pos.size(), out_data.end(), out_right_pos.begin());

    l_idx.resize(out_left_pos.size());
    r_idx.resize(out_left_pos.size());
    thrust::fill(l_idx.begin(), l_idx.end(), -1);
    thrust::fill(r_idx.begin(), r_idx.end(), -1);
    thrust::gather_if(out_left_pos.begin(), out_left_pos.end(), out_left_pos.begin(), dl.begin(), l_idx.begin(), non_negative<int>());
    thrust::gather_if(out_right_pos.begin(), out_right_pos.end(), out_right_pos.begin(), dr.begin(), r_idx.begin(), non_negative<int>());

    return err;
}

template <typename T>
gdf_error
call_gdf_multi_column_test(
        std::array<thrust::device_vector<T>, 3> &l,
        std::array<thrust::device_vector<T>, 3> &r,
        thrust::device_vector<int> &out_left_pos,
        thrust::device_vector<int> &out_right_pos,
        const int index) {
    std::vector<T> l0{0, 0, 4, 5, 5};
    std::vector<T> l1{1, 2, 2, 3, 4};
    std::vector<T> l2{1, 1, 3, 1, 2};
    std::vector<T> r0{0, 0, 2, 3, 5};
    std::vector<T> r1{1, 2, 3, 3, 4};
    std::vector<T> r2{3, 3, 2, 1, 1};

    thrust::device_vector<T> dl0 = l0; thrust::swap(dl0, l[0]);
    thrust::device_vector<T> dl1 = l1; thrust::swap(dl1, l[1]);
    thrust::device_vector<T> dl2 = l2; thrust::swap(dl2, l[2]);
    thrust::device_vector<T> dr0 = r0; thrust::swap(dr0, r[0]);
    thrust::device_vector<T> dr1 = r1; thrust::swap(dr1, r[1]);
    thrust::device_vector<T> dr2 = r2; thrust::swap(dr2, r[2]);

    gdf_column gdl0 = create_gdf_column(l[0]);
    gdf_column gdl1 = create_gdf_column(l[1]);
    gdf_column gdl2 = create_gdf_column(l[2]);

    gdf_column gdr0 = create_gdf_column(r[0]);
    gdf_column gdr1 = create_gdf_column(r[1]);
    gdf_column gdr2 = create_gdf_column(r[2]);

    gdf_column* gl[3] = {&gdl0, &gdl1, &gdl2};
    gdf_column* gr[3] = {&gdr0, &gdr1, &gdr2};
    gdf_join_result_type *out;
    gdf_error err = gdf_multi_left_join_generic(index, gl, gr, &out);

    size_t len = gdf_join_result_size(out);
    size_t hlen = len/2;
    int* out_ptr = reinterpret_cast<int*>(gdf_join_result_data(out));
    thrust::device_vector<int> out_data(out_ptr, out_ptr + len);

    thrust::sort_by_key(out_data.begin() + hlen, out_data.end(), out_data.begin());
    thrust::sort_by_key(out_data.begin(), out_data.begin() + hlen, out_data.begin() + hlen);
    out_left_pos.resize(hlen);
    out_right_pos.resize(hlen);

    thrust::copy(out_data.begin(), out_data.begin() + out_left_pos.size(), out_left_pos.begin());
    thrust::copy(out_data.begin() + out_right_pos.size(), out_data.end(), out_right_pos.begin());
    return err;
}

template <typename T>
void gdf_multi_left_join_test_index1(void) {
    std::array<thrust::device_vector<T>, 3> l;
    std::array<thrust::device_vector<T>, 3> r;
    thrust::device_vector<int> l_pos;
    thrust::device_vector<int> r_pos;
    auto err = call_gdf_multi_column_test(l, r, l_pos, r_pos, 1);
    thrust::device_vector<T> map_out(l_pos.size(), -1);

    EXPECT_THAT(host_vec(l_pos), ElementsAre(0, 0, 1, 1, 2, 3, 4));
    EXPECT_THAT(host_vec(r_pos), ElementsAre(0, 1, 0, 1, -1, 4, 4));

    thrust::gather_if(l_pos.begin(), l_pos.end(), l_pos.begin(), l[0].begin(), map_out.begin(),
            non_negative<int>());
    EXPECT_THAT(host_vec(map_out), ElementsAre(0, 0, 0, 0, 4, 5, 5));

    ASSERT_EQ(err, GDF_SUCCESS);
}

TEST(gdf_multi_left_join_TEST, i8_index1) {
    gdf_multi_left_join_test_index1<int8_t>();
}

TEST(gdf_multi_left_join_TEST, i32_index1) {
    gdf_multi_left_join_test_index1<int32_t>();
}

TEST(gdf_multi_left_join_TEST, i64_index1) {
    gdf_multi_left_join_test_index1<int64_t>();
}

TEST(gdf_multi_left_join_TEST, f32_index1) {
    gdf_multi_left_join_test_index1<float>();
}

TEST(gdf_multi_left_join_TEST, f64_index1) {
    gdf_multi_left_join_test_index1<double>();
}

template <typename T>
void gdf_multi_left_join_test_index2(void) {
    std::array<thrust::device_vector<T>, 3> l;
    std::array<thrust::device_vector<T>, 3> r;
    thrust::device_vector<int> l_pos;
    thrust::device_vector<int> r_pos;
    auto err = call_gdf_multi_column_test(l, r, l_pos, r_pos, 2);
    thrust::device_vector<T> map_out(l_pos.size());

    EXPECT_THAT(host_vec(l_pos), ElementsAre(0, 1, 2, 3, 4));

    {
        thrust::fill(map_out.begin(), map_out.end(), -1);
        thrust::gather_if(l_pos.begin(), l_pos.end(), l_pos.begin(), l[0].begin(), map_out.begin(),
                non_negative<int>());
        EXPECT_THAT(host_vec(map_out), ElementsAre(0, 0, 4, 5, 5));
    }

    {
        thrust::fill(map_out.begin(), map_out.end(), -1);
        thrust::gather_if(l_pos.begin(), l_pos.end(), l_pos.begin(), l[1].begin(), map_out.begin(),
                non_negative<int>());
        EXPECT_THAT(host_vec(map_out), ElementsAre(1, 2, 2, 3, 4));
    }

    ASSERT_EQ(err, GDF_SUCCESS);
}

TEST(gdf_multi_left_join_TEST, i8_index2) {
    gdf_multi_left_join_test_index2<int8_t>();
}

TEST(gdf_multi_left_join_TEST, i32_index2) {
    gdf_multi_left_join_test_index2<int32_t>();
}

TEST(gdf_multi_left_join_TEST, i64_index2) {
    gdf_multi_left_join_test_index2<int64_t>();
}

TEST(gdf_multi_left_join_TEST, f32_index2) {
    gdf_multi_left_join_test_index2<float>();
}

TEST(gdf_multi_left_join_TEST, f64_index2) {
    gdf_multi_left_join_test_index2<double>();
}

template <typename T>
void gdf_multi_left_join_test_index3(void) {
    std::array<thrust::device_vector<T>, 3> l;
    std::array<thrust::device_vector<T>, 3> r;
    thrust::device_vector<int> l_pos;
    thrust::device_vector<int> r_pos;
    auto err = call_gdf_multi_column_test(l, r, l_pos, r_pos, 3);
    thrust::device_vector<T> map_out(l_pos.size());

    EXPECT_THAT(host_vec(l_pos), ElementsAre(0, 1, 2, 3, 4));

    {
        thrust::fill(map_out.begin(), map_out.end(), -1);
        thrust::gather_if(l_pos.begin(), l_pos.end(), l_pos.begin(), l[0].begin(), map_out.begin(),
                non_negative<int>());
        EXPECT_THAT(host_vec(map_out), ElementsAre(0, 0, 4, 5, 5));
    }

    {
        thrust::fill(map_out.begin(), map_out.end(), -1);
        thrust::gather_if(l_pos.begin(), l_pos.end(), l_pos.begin(), l[1].begin(), map_out.begin(),
                non_negative<int>());
        EXPECT_THAT(host_vec(map_out), ElementsAre(1, 2, 2, 3, 4));
    }

    {
        thrust::fill(map_out.begin(), map_out.end(), -1);
        thrust::gather_if(l_pos.begin(), l_pos.end(), l_pos.begin(), l[2].begin(), map_out.begin(),
                non_negative<int>());
        EXPECT_THAT(host_vec(map_out), ElementsAre(1, 1, 3, 1, 2));
    }

    ASSERT_EQ(err, GDF_SUCCESS);
}

TEST(gdf_multi_left_join_TEST, i8_index3) {
    gdf_multi_left_join_test_index3<int8_t>();
}

TEST(gdf_multi_left_join_TEST, i32_index3) {
    gdf_multi_left_join_test_index3<int32_t>();
}

TEST(gdf_multi_left_join_TEST, i64_index3) {
    gdf_multi_left_join_test_index3<int64_t>();
}

TEST(gdf_multi_left_join_TEST, f32_index3) {
    gdf_multi_left_join_test_index3<float>();
}

TEST(gdf_multi_left_join_TEST, f64_index3) {
    gdf_multi_left_join_test_index3<double>();
}

template <typename T>
void gdf_inner_join_test(void) {
    std::vector<T> l{0, 0, 1, 2, 3};
    std::vector<T> r{0, 1, 2, 2, 3};
    thrust::device_vector<T> l_idx, r_idx;
    thrust::device_vector<int> l_pos, r_pos;

    auto err = call_gdf_single_column_test(l, r, l_pos, r_pos, l_idx, r_idx, gdf_inner_join_generic);

    EXPECT_THAT(host_vec(l_idx), ElementsAreArray(host_vec(r_idx)));
    EXPECT_THAT(host_vec(l_pos), ElementsAre(0, 1, 2, 3, 3, 4));
    EXPECT_THAT(host_vec(r_pos), ElementsAre(0, 0, 1, 2, 3, 4));

    ASSERT_EQ(err, GDF_SUCCESS);
}

TEST(join_TEST, gdf_inner_join_i8) {
    gdf_inner_join_test<int8_t>();
}

//TEST(join_TEST, gdf_inner_join_i16) {
//    gdf_inner_join_test<int16_t>();
//}

TEST(join_TEST, gdf_inner_join_i32) {
    gdf_inner_join_test<int32_t>();
}

TEST(join_TEST, gdf_inner_join_i64) {
    gdf_inner_join_test<int64_t>();
}

TEST(join_TEST, gdf_inner_join_f32) {
    gdf_inner_join_test<float>();
}

TEST(join_TEST, gdf_inner_join_f64) {
    gdf_inner_join_test<double>();
}

template <typename T>
void gdf_left_join_test(void) {
    std::vector<T> l{0, 0, 4, 5, 5};
    std::vector<T> r{0, 0, 2, 3, 5};
    thrust::device_vector<T> l_idx, r_idx;
    thrust::device_vector<int> l_pos, r_pos;

    auto err = call_gdf_single_column_test(l, r, l_pos, r_pos, l_idx, r_idx, gdf_left_join_generic);

    EXPECT_THAT(host_vec(l_idx), ElementsAre(0, 0, 0, 0, 4, 5, 5));
    EXPECT_THAT(host_vec(l_pos), ElementsAre(0, 0, 1, 1, 2, 3, 4));
    EXPECT_THAT(host_vec(r_pos), ElementsAre(0, 1, 0, 1, -1, 4, 4));

    ASSERT_EQ(err, GDF_SUCCESS);
}

TEST(join_TEST, gdf_left_join_i8) {
    gdf_left_join_test<int8_t>();
}

//TEST(join_TEST, gdf_left_join_i16) {
//    gdf_left_join_test<int16_t>();
//}

TEST(join_TEST, gdf_left_join_i32) {
    gdf_left_join_test<int32_t>();
}

TEST(join_TEST, gdf_left_join_i64) {
    gdf_left_join_test<int64_t>();
}

TEST(join_TEST, gdf_left_join_f32) {
    gdf_left_join_test<float>();
}

TEST(join_TEST, gdf_left_join_f64) {
    gdf_left_join_test<double>();
}

template <typename T>
void gdf_outer_join_test(void) {
    std::vector<T> l{0, 0, 4, 5, 5};
    std::vector<T> r{0, 0, 2, 3, 5};
    thrust::device_vector<T> l_idx, r_idx;
    thrust::device_vector<int> l_pos, r_pos;

    auto err = call_gdf_single_column_test(l, r, l_pos, r_pos, l_idx, r_idx, gdf_outer_join_generic);

    EXPECT_THAT(host_vec(l_idx), ElementsAre(-1, -1, 0, 0, 0, 0,  4, 5, 5));
    EXPECT_THAT(host_vec(r_idx), ElementsAre( 2,  3, 0, 0, 0, 0, -1, 5, 5));
    EXPECT_THAT(host_vec(l_pos), ElementsAre(-1, -1, 0, 0, 1, 1,  2, 3, 4));
    EXPECT_THAT(host_vec(r_pos), ElementsAre( 2,  3, 0, 1, 0, 1, -1, 4, 4));

    ASSERT_EQ(err, GDF_SUCCESS);
}

TEST(join_TEST, gdf_outer_join_i8) {
    gdf_outer_join_test<int8_t>();
}

//TEST(join_TEST, gdf_outer_join_i16) {
//    gdf_outer_join_test<int16_t>();
//}

TEST(join_TEST, gdf_outer_join_i32) {
    gdf_outer_join_test<int32_t>();
}

TEST(join_TEST, gdf_outer_join_i64) {
    gdf_outer_join_test<int64_t>();
}

TEST(join_TEST, gdf_outer_join_f32) {
    gdf_outer_join_test<float>();
}

TEST(join_TEST, gdf_outer_join_f64) {
    gdf_outer_join_test<double>();
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
