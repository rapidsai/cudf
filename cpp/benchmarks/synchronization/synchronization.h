// Google Benchmark library
#include <benchmark/benchmark.h>

struct cuda_event_timer {
  cudaEvent_t start, stop;
  
  benchmark::State* p_state;

  cuda_event_timer(benchmark::State& state): p_state(&state) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }
  
  cuda_event_timer(){
    assert(false);
  }
  
  ~cuda_event_timer(){  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    p_state->SetIterationTime(milliseconds/1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};

