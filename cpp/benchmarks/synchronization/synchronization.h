// Google Benchmark library
#include <benchmark/benchmark.h>

class CudaState;

class CudaStateIterator {
  private:
    
    benchmark::State::StateIterator state_iter;
    
    CudaState* parent;

  public:
    // The following is for operator* which is used in the range-based for loop.
    // This is just a dummy convention to suppress compiler warnings.
    struct __attribute__((unused)) Value {};
    Value operator*() const { return Value(); }
    
    CudaStateIterator(CudaState* cuda_state, benchmark::State::StateIterator state_iter_): parent(cuda_state), state_iter(state_iter_) { }

    CudaStateIterator& operator++();
    
    bool operator!=(CudaStateIterator const& cuda_state_iter);
};

class CudaState {
  private:
   
    friend class CudaStateIterator;

    cudaEvent_t start, stop;
    bool started;
    benchmark::State* p_state; 
  
  public:

    CudaState(benchmark::State& state): p_state(&state), started(false) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
    }
    
    ~CudaState(){
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }   
   
    CudaStateIterator begin() {
      return CudaStateIterator(this, p_state->begin());
    }
 
    CudaStateIterator end() {
      return CudaStateIterator(this, p_state->end());
    }
    
    bool KeepRunning() {
      
      if(started){
        // record stop and set user defined (CUDA) time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        p_state->SetIterationTime(milliseconds/1000.0f);
      }else{
        started = true;
      }
      
      bool keep_running = p_state->KeepRunning();
      
      if(keep_running){
        // record start since we are going to run the iteration.
        CUDA_TRY(cudaEventRecord(start));
      }else{
        started = false;
      }
      
      return keep_running;
    }
}; 

CudaStateIterator& CudaStateIterator::operator++() {
  
  // stop the timer and record
  cudaEventRecord(parent->stop);
  cudaEventSynchronize(parent->stop);
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, parent->start, parent->stop);
  parent->p_state->SetIterationTime(milliseconds/1000.0f);
  // printf("timer stop: %f\n", milliseconds);

  ++state_iter;
  return *this;
}

bool CudaStateIterator::operator!=(CudaStateIterator const& cuda_state_iter) {
  bool keep_running = state_iter != cuda_state_iter.state_iter;
  if(keep_running){
    // start the timer
    // printf("timer start\n");
    CUDA_TRY(cudaEventRecord(parent->start));
  }
  return keep_running;
}

