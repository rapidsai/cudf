// Google Benchmark library
#include <benchmark/benchmark.h>

struct cuda_event_timer {
  cudaEvent_t start, stop;
  
  benchmark::State* p_state;

  cuda_event_timer(benchmark::State& state): p_state(&state) {
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&stop));
    CUDA_TRY(cudaEventRecord(start));
  }
  
  cuda_event_timer(){
    assert(false);
  }
  
  ~cuda_event_timer(){  
    CUDA_TRY(cudaEventRecord(stop));
    CUDA_TRY(cudaEventSynchronize(stop));
    
    float milliseconds = 0.0f;
    CUDA_TRY(cudaEventElapsedTime(&milliseconds, start, stop));
    p_state->SetIterationTime(milliseconds/1000.0f);
    
    CUDA_TRY(cudaEventDestroy(start));
    CUDA_TRY(cudaEventDestroy(stop));
  }
};

