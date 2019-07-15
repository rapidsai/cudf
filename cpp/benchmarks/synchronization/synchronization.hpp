// Google Benchmark library
#include <benchmark/benchmark.h>

#include <tests/utilities/cudf_test_utils.cuh>

class cuda_event_timer {
 
public:
  
  // The c'tor clears the L2$ by cudaMemset a buffer of L2$ size
  // and start the timer. 
  cuda_event_timer(benchmark::State& state, cudaStream_t stream_ = 0);  
  
  cuda_event_timer() = delete;  
  
  // The d'tor stops the timer and perform a synchroniazation. If 
  // a benchmark::State object is provided to the c'tor its time 
  // will be set to the value given `cudaEventElapsedTime`.
  ~cuda_event_timer();

private:
 
  cudaEvent_t start;
  cudaEvent_t stop;
  
  int l2_cache_bytes = 0;
  int current_device = 0;
  int* l2_cache_buffer = nullptr;

  cudaStream_t stream;

  benchmark::State* p_state = nullptr;

};

