// Driver-only model of the CI crash (no UCX, rapidsmpf or cudf).
//
// What happens in CI: a rapidsmpf shuffler thread calls ucp_tag_send_nbx on a
// small (384-1024 B) buffer allocated with plain cudaMalloc (rmm cuda_memory_resource).
// UCX 1.22 (uct_cuda_copy_md_mem_query -> detect_mem_flags -> get_dmabuf) then does:
//     cuMemGetAddressRange(&base, &size, ptr)            // -> the allocation itself
//     start = align_down(base, 4096); end = align_up(base + size, 4096)
//     cuMemGetHandleForAddressRange(&fd, start, end - start, DMA_BUF_FD, 0)
// The page-aligned range usually covers *neighbouring* small allocations, which
// other threads are concurrently allocating/freeing.
//
// This program: "sender" threads own a live small buffer, run the UCX sequence on
// it, then free it; "churn" threads allocate/free neighbouring small buffers.
//
// usage: dmabuf_race [seconds=20] [senders=2] [churners=4] [alloc=malloc|async]
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PAGE 4096ull
#define NSLOT 2048

static void *slots[NSLOT];
static atomic_int stop;
static CUcontext ctx;
static int use_async;
static atomic_long ok, errs, queries_spanning_neighbours;

static void *xalloc(size_t n, cudaStream_t s) {
    void *p = NULL;
    cudaError_t e = use_async ? cudaMallocAsync(&p, n, s) : cudaMalloc(&p, n);
    return e == cudaSuccess ? p : NULL;
}
static void xfree(void *p, cudaStream_t s) {
    if (use_async) cudaFreeAsync(p, s); else cudaFree(p);
}
// 384..1024 B in 128 B steps, like the sizes seen in the CI backtraces
static size_t small_size(unsigned *seed) { return 384 + 128 * (rand_r(seed) % 6); }

static void *churn(void *a) {
    cuCtxSetCurrent(ctx);
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    unsigned seed = (unsigned)(long)a * 7919 + 1;
    while (!atomic_load(&stop)) {
        int i = rand_r(&seed) % NSLOT;
        void *p = xalloc(small_size(&seed), s);
        void *prev = __atomic_exchange_n(&slots[i], p, __ATOMIC_ACQ_REL);
        if (prev) xfree(prev, s);
        if (use_async && (rand_r(&seed) & 63) == 0) cudaStreamSynchronize(s);
    }
    return NULL;
}

static void *sender(void *a) {
    cuCtxSetCurrent(ctx);
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    unsigned seed = (unsigned)(long)a * 104729 + 3;
    while (!atomic_load(&stop)) {
        void *buf = xalloc(small_size(&seed), s);
        if (!buf) continue;
        if (use_async) cudaStreamSynchronize(s);

        // --- UCX's sequence ---
        CUdeviceptr base; size_t size;
        if (cuMemGetAddressRange(&base, &size, (CUdeviceptr)buf) == CUDA_SUCCESS) {
            CUdeviceptr start = base & ~(PAGE - 1);
            CUdeviceptr end = (base + size + PAGE - 1) & ~(PAGE - 1);
            if (start != base || end != base + size) atomic_fetch_add(&queries_spanning_neighbours, 1);
            int fd = -1;
            CUresult r = cuMemGetHandleForAddressRange(&fd, start, end - start,
                                                       CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
            if (r == CUDA_SUCCESS) { close(fd); atomic_fetch_add(&ok, 1); }
            else atomic_fetch_add(&errs, 1);
        }
        xfree(buf, s);
    }
    return NULL;
}

int main(int argc, char **argv) {
    int secs = argc > 1 ? atoi(argv[1]) : 20;
    int ns = argc > 2 ? atoi(argv[2]) : 2, nc = argc > 3 ? atoi(argv[3]) : 4;
    use_async = argc > 4 && !strcmp(argv[4], "async");
    cudaSetDevice(0); cudaFree(0); cuCtxGetCurrent(&ctx);
    pthread_t t[64]; int n = 0;
    for (long i = 0; i < nc; i++) pthread_create(&t[n++], NULL, churn, (void *)i);
    for (long i = 0; i < ns; i++) pthread_create(&t[n++], NULL, sender, (void *)i);
    sleep(secs); atomic_store(&stop, 1);
    for (int i = 0; i < n; i++) pthread_join(t[i], NULL);
    printf("done (%s): ok=%ld err=%ld, queries wider than the allocation=%ld\n",
           use_async ? "async" : "malloc", ok, errs, queries_spanning_neighbours);
    return 0;
}
