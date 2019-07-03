#include <benchmark/benchmark.h>

namespace cudf {

class benchmark : public ::benchmark::Fixture {
public:
  virtual void SetUp(const ::benchmark::State& state) {
    rmmOptions_t options{PoolAllocation, 0, false};
    rmmInitialize(&options); 
  }

  virtual void TearDown(const ::benchmark::State& state) {
    rmmFinalize();
  }

  // eliminate partial override warnings (see benchmark/benchmark.h)
  virtual void SetUp(::benchmark::State& st) { 
    SetUp(const_cast<const ::benchmark::State&>(st)); 
  }
  virtual void TearDown(::benchmark::State& st) { 
    TearDown(const_cast<const ::benchmark::State&>(st)); 
  }
};

};
