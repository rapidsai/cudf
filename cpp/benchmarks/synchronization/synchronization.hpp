// Google Benchmark library
#include <benchmark/benchmark.h>

#include <tests/utilities/cudf_test_utils.cuh>

class cuda_event_timer {
 
public:
  
  // The two c'tor clear the L2$ by cudaMemset a buffer of L2$ size
  // and start the timer. 
  cuda_event_timer(benchmark::State& state, cudaStream_t stream_ = 0);  
  
  cuda_event_timer(cudaStream_t stream_ = 0);  
  
  // The d'tor stops the timer and perform a synchroniazation. If 
  // a benchmark::State object is provided to the c'tor its time 
  // will be set to the value given `cudaEventElapsedTime`.
  ~cuda_event_timer();

private:
 
  void init();

  cudaEvent_t start, stop;
  
  static int l2_cache_bytes;
  static int current_device;
  static int* l2_cache_buffer;
 
  cudaStream_t stream;

  benchmark::State* p_state = nullptr;

};

