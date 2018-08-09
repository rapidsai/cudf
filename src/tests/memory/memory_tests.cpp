#include "gtest/gtest.h"
#include "rmm.h"

// Helper macros to simplify testing for success or failure
#define ASSERT_SUCCESS(res) ASSERT_EQ(RMM_SUCCESS, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(RMM_SUCCESS, (res));

/// Helper class for similar tests
class MemoryManagerTest : public ::testing::Test {
protected:
  virtual void SetUp() {
  	ASSERT_SUCCESS(rmmInitialize());

  	cudaError_t res = cudaStreamCreate(&stream);
  	ASSERT_EQ(cudaSuccess, res);
  }

  virtual void TearDown() {
  	cudaError_t res = cudaStreamDestroy(stream);
  	ASSERT_EQ(cudaSuccess, res);

  	ASSERT_SUCCESS(rmmFinalize());
  }	

  cudaStream_t stream;

  // some useful allocation sizes
  const size_t size_word = 4;
  const size_t size_kb = size_t(1)<<10;
  const size_t size_mb = size_t(1)<<20;
  const size_t size_gb = size_t(1)<<30;
  const size_t size_tb = size_t(1)<<40;
  const size_t size_pb = size_t(1)<<50;
};

// Init / Finalize tests

TEST_F(MemoryManagerTest, Initialize) {
	ASSERT_SUCCESS(rmmInitialize());
}

TEST_F(MemoryManagerTest, Finalize) {
	ASSERT_SUCCESS(rmmFinalize());
}

// zero size tests

TEST_F(MemoryManagerTest, AllocateZeroBytes) {
	char *a = 0;
	ASSERT_SUCCESS(rmmAlloc((void**)&a, 0, stream));
}

TEST_F(MemoryManagerTest, NullPtrAllocateZeroBytes) {
	ASSERT_SUCCESS(rmmAlloc(0, 0, stream));
}

// Bad argument tests

TEST_F(MemoryManagerTest, NullPtrInvalidArgument) {
	rmmError_t res = rmmAlloc(0, 4, stream);
	ASSERT_FAILURE(res);
	ASSERT_EQ(RMM_ERROR_INVALID_ARGUMENT, res);
}

// Simple allocation / free tests

TEST_F(MemoryManagerTest, AllocateWord) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_kb, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, AllocateKB) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_word, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, AllocateMB) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_mb, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, AllocateGB) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_gb, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, AllocateTB) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_tb, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, AllocateTooMuch) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_pb, stream);
	ASSERT_FAILURE(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, FreeZero) {
	ASSERT_SUCCESS(rmmFree(0, stream));
}

// Reallocation tests

TEST_F(MemoryManagerTest, ReallocateSmaller) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_mb, stream);
	ASSERT_SUCCESS(res);
	res = rmmRealloc((void**)&a, size_mb / 2, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, ReallocateMuchSmaller) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_gb, stream);
	ASSERT_SUCCESS(res);
	res = rmmRealloc((void**)&a, size_kb, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}


TEST_F(MemoryManagerTest, ReallocateLarger) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_mb, stream);
	ASSERT_SUCCESS(res);
	res = rmmRealloc((void**)&a, size_mb * 2, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, ReallocateMuchLarger) {
	char *a = 0;
	rmmError_t res = rmmAlloc((void**)&a, size_kb, stream);
	ASSERT_SUCCESS(res);
	res = rmmRealloc((void**)&a, size_gb, stream);
	ASSERT_SUCCESS(res);
	res = rmmFree(a, stream);
	ASSERT_SUCCESS(res);
}

TEST_F(MemoryManagerTest, GetInfo) {
	size_t freeBefore = 0, totalBefore = 0;
	rmmError_t res = rmmGetInfo(&freeBefore, &totalBefore, stream);
	ASSERT_SUCCESS(res);
	ASSERT_NE(freeBefore, 0);
	ASSERT_NE(totalBefore, 0);

	char *a = 0;
	size_t sz = size_gb / 2;
	res = rmmAlloc((void**)&a, sz, stream);
	ASSERT_SUCCESS(res);

	// make sure the available free memory goes down after an allocation
	size_t freeAfter = 0, totalAfter = 0;
	res = rmmGetInfo(&freeAfter, &totalAfter, stream);
	ASSERT_SUCCESS(res);
	ASSERT_EQ(totalAfter, totalBefore);
	ASSERT_LE(freeAfter, freeBefore);
}
