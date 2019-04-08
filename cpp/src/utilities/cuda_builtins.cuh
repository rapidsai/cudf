/**
 * This code in this this file is (C) Eyal Rozenberg and CWI Amsterdam
 * under terms of the BSD 3-Clause license. See:
 * https://github.com/eyalroz/libgiddy/blob/master/LICENSE
 *
 * Talk to Eyal <eyalroz@blazingdb.com> about sublicensing/relicensing
 */


#ifndef CUDA_BUILTINS_CUH_
#define CUDA_BUILTINS_CUH_

#include <cuda_runtime.h>

#if (__CUDACC_VER_MAJOR__ < 9)
#error "CUDA 9.0 or higher is required"
#endif


#define __fd__ __forceinline__ __device__

using lane_mask_t = unsigned;


enum : lane_mask_t { full_warp_mask = 0xFFFFFFFF };

#ifndef WARP_SIZE
enum : unsigned { warp_size = 32 };
    // Why have this when we have warp_size? Because the latter is a built-in variable,
    // not a compile-time constant, for some strange reason.
#endif


namespace builtins {

template <typename T> __fd__ T minimum(T x, T y);
template <> __fd__ int                 minimum<int               >(int x, int y)                               { return min(x,y);    }
template <> __fd__ unsigned int        minimum<unsigned          >(unsigned int x, unsigned int y)             { return umin(x,y);   }
template <> __fd__ long                minimum<long              >(long x, long y)                             { return llmin(x,y);  }
template <> __fd__ unsigned long       minimum<unsigned long     >(unsigned long x, unsigned long y)           { return ullmin(x,y); }
template <> __fd__ long long           minimum< long long        >(long long x, long long y)                   { return llmin(x,y);  }
template <> __fd__ unsigned long long  minimum<unsigned long long>(unsigned long long x, unsigned long long y) { return ullmin(x,y); }
template <> __fd__ float               minimum<float             >(float x, float y)                           { return fminf(x,y);  }
template <> __fd__ double              minimum<double            >(double x, double y)                         { return fmin(x,y);   }

template <typename T> __fd__ T maximum(T x, T y);
template <> __fd__ int                 maximum<int               >(int x, int y)                               { return max(x,y);    }
template <> __fd__ unsigned int        maximum<unsigned          >(unsigned int x, unsigned int y)             { return umax(x,y);   }
template <> __fd__ long                maximum<long              >(long x, long y)                             { return llmax(x,y);  }
template <> __fd__ unsigned long       maximum<unsigned long     >(unsigned long x, unsigned long y)           { return ullmax(x,y); }
template <> __fd__ long long           maximum< long long        >(long long x, long long y)                   { return llmax(x,y);  }
template <> __fd__ unsigned long long  maximum<unsigned long long>(unsigned long long x, unsigned long long y) { return ullmax(x,y); }
template <> __fd__ float               maximum<float             >(float x, float y)                           { return fmaxf(x,y);  }
template <> __fd__ double              maximum<double            >(double x, double y)                         { return fmax(x,y);   }

template <typename T> __fd__ int count_leading_zeros(T x);
template <> __fd__ int count_leading_zeros<int               >(int x)                { return __clz(x);   }
template <> __fd__ int count_leading_zeros<unsigned          >(unsigned int x)       { return __clz(x);   }
template <> __fd__ int count_leading_zeros<long long         >(long long x)          { return __clzll(x); }
template <> __fd__ int count_leading_zeros<unsigned long long>(unsigned long long x) { return __clzll(x); }

template <typename T> __fd__ int population_count(T x);
template <          > __fd__ int population_count<unsigned          >(unsigned int       x) { return __popc(x);   }
template <          > __fd__ int population_count<unsigned long long>(unsigned long long x) { return __popcll(x); }

__fd__ lane_mask_t warp_ballot(unsigned lane_mask, int cond) { return __ballot_sync(lane_mask, cond); }
// Not really a built-in...
__fd__ lane_mask_t warp_ballot(int cond) { return warp_ballot(full_warp_mask, cond); }

namespace shuffle {

template <typename T> __fd__ T arbitrary_sync(unsigned int mask, T x, int source_lane, int width = warp_size) { return __shfl_sync(mask, x, source_lane, width);   }
template <typename T> __fd__ T down_sync(unsigned int mask, T x, unsigned delta, int width = warp_size)       { return __shfl_down_sync(mask, x, delta, width);    }
template <typename T> __fd__ T up_sync(unsigned int mask, T x, unsigned delta, int width = warp_size)         { return __shfl_up_sync(mask, x, delta, width);      }
template <typename T> __fd__ T xor_sync(unsigned int mask, T x, int lane_mask, int width = warp_size)         { return __shfl_xor_sync(mask, x, lane_mask, width); }

// Not really built-ins...
template <typename T> __fd__ T arbitrary(T x, int source_lane, int width = warp_size) { return arbitrary_sync<T>(::full_warp_mask, x, source_lane, width); }
template <typename T> __fd__ T down(T x, unsigned delta, int width = warp_size)       { return down_sync<T>(::full_warp_mask, x, delta, width);            }
template <typename T> __fd__ T up(T x, unsigned delta, int width = warp_size)         { return up_sync<T>(::full_warp_mask, x, delta, width);              }
template <typename T> __fd__ T xor_(T x, int lane_mask, int width = warp_size)        { return xor_sync<T>(::full_warp_mask, x, lane_mask, width);         }

}

} // namespace builtins


#undef __fd__

#endif // CUDA_BUILTINS_CUH_
