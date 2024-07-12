#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>
#include <src/io/utilities/multibuffer_memset.hpp>
#include <thrust/iterator/transform_iterator.h>
#include <type_traits>
#include <cudf/io/parquet.hpp>

template <typename T>
struct MultiBufferTestIntegral : public cudf::test::BaseFixture {};

TEST(MultiBufferTestIntegral, BasicTest)
{
    long NUM_BUFS = 1;
    std::vector<long> BUF_SIZES{600};

    // Device init
    std::vector<cudf::device_span<uint8_t>> bufs;
    auto stream = cudf::get_default_stream();
    auto _mr = rmm::mr::get_current_device_resource();
    for (int i = 0; i < NUM_BUFS; i++) {
        rmm::device_uvector<uint8_t> temp(BUF_SIZES[i], stream, _mr);
        bufs.push_back(cudf::device_span<uint8_t>(temp));
    }
    multibuffer_memset(bufs, 0, stream, _mr);

    // Compare
    for (int i = 0; i < NUM_BUFS; i++) {
        std::vector<uint8_t> temp(BUF_SIZES[i]);
        cudf::host_span<uint8_t> host(temp);
        CUDF_CUDA_TRY(cudaMemcpy(host.data(), bufs[i].data(), BUF_SIZES[i], cudaMemcpyDefault));
        for (int i = 0; i < BUF_SIZES[i]; i++) {
            EXPECT_EQ(host[i], 0);
        }
    }
}


TEST(MultiBufferTestIntegral, BasicTest2)
{
    long NUM_BUFS = 1;
    std::vector<long> BUF_SIZES{131073};

    // Device init
    std::vector<cudf::device_span<uint8_t>> bufs;
    auto stream = cudf::get_default_stream();
    auto _mr = rmm::mr::get_current_device_resource();
    for (int i = 0; i < NUM_BUFS; i++) {
        rmm::device_uvector<uint8_t> temp(BUF_SIZES[i], stream, _mr);
        bufs.push_back(cudf::device_span<uint8_t>(temp));
    }
    multibuffer_memset(bufs, 1, stream, _mr);

    // Compare
    for (int i = 0; i < NUM_BUFS; i++) {
        std::vector<uint8_t> temp(BUF_SIZES[i]);
        cudf::host_span<uint8_t> host(temp);
        CUDF_CUDA_TRY(cudaMemcpy(host.data(), bufs[i].data(), BUF_SIZES[i], cudaMemcpyDefault));
        for (int i = 0; i < BUF_SIZES[i]; i++) {
            EXPECT_EQ(host[i], 1);
        }
    }
}

TEST(MultiBufferTestIntegral, BasicTest3)
{
    long NUM_BUFS = 8;
    std::vector<long> BUF_SIZES{100, 0, 300, 400, 500, 600, 700, 800};

    // Device init
    std::vector<cudf::device_span<uint8_t>> bufs;
    auto stream = cudf::get_default_stream();
    auto _mr = rmm::mr::get_current_device_resource();
    for (int i = 0; i < NUM_BUFS; i++) {
        rmm::device_uvector<uint8_t> temp(BUF_SIZES[i], stream, _mr);
        bufs.push_back(cudf::device_span<uint8_t>(temp));
    }
    multibuffer_memset(bufs, 2, stream, _mr);

    // Compare
    for (int i = 0; i < NUM_BUFS; i++) {
        std::vector<uint8_t> temp(BUF_SIZES[i]);
        cudf::host_span<uint8_t> host(temp);
        CUDF_CUDA_TRY(cudaMemcpy(host.data(), bufs[i].data(), BUF_SIZES[i], cudaMemcpyDefault));
        for (int i = 0; i < BUF_SIZES[i]; i++) {
            EXPECT_EQ(host[i], 2);
        }
    }
}

TEST(MultiBufferTestIntegral, BasicTest4)
{
    long NUM_BUFS = 8;
    std::vector<long> BUF_SIZES{131073, 200, 160000, 300000, 500000, 600, 131700, 800};

    // Device init
    std::vector<cudf::device_span<uint8_t>> bufs;
    auto stream = cudf::get_default_stream();
    auto _mr = rmm::mr::get_current_device_resource();
    for (int i = 0; i < NUM_BUFS; i++) {
        rmm::device_uvector<uint8_t> temp(BUF_SIZES[i], stream, _mr);
        bufs.push_back(cudf::device_span<uint8_t>(temp));
    }
    multibuffer_memset(bufs, 3, stream, _mr);

    // Compare
    for (int i = 0; i < NUM_BUFS; i++) {
        std::vector<uint8_t> temp(BUF_SIZES[i]);
        cudf::host_span<uint8_t> host(temp);
        CUDF_CUDA_TRY(cudaMemcpy(host.data(), bufs[i].data(), BUF_SIZES[i], cudaMemcpyDefault));
        for (int i = 0; i < BUF_SIZES[i]; i++) {
            EXPECT_EQ(host[i], 3);
        }
    }
}