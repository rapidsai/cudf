#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdio>
#include <random>
#include <vector>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

//
// cd ../build
// g++ -std=c++11 ../tests/normalize_spaces.cpp -I/usr/local/cuda/include -L. -lNVStrings -lNVText
// -o normalize_spaces -Wl,-rpath,.:
//

double GetTime()
{
  timeval tv;
  gettimeofday(&tv, NULL);
  return (double)(tv.tv_sec * 1000000 + tv.tv_usec) / 1000000.0;
}

std::vector<const char*> hstrs{
  " the\t quick brown fox   jumps over the lazy dog",
  "the fat cat lays next to the other\f accénted cat\r",
  "a slow moving turtlé cannot catch the bird\n",
  "which can be composéd     together to form  a more  complete",
  "    thé   result  does \f\tnot\f include the  value in   the sum in  ",
  "",
  "should be no change to this string of words"};

void test_regex()
{
  // std::random_device rd;
  // std::mt19937 mt(rd());
  // std::uniform_int_distribution<int> dist(0,hstrs.size());
  std::vector<const char*> data_ptrs;
  for (int idx = 0; idx < 100000000; ++idx)
    data_ptrs.push_back(hstrs[idx % hstrs.size()]);  // dist(mt)
  NVStrings* strs = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size());
  printf("strings(%d): (%ld bytes)\n", strs->size(), strs->memsize());
  strs->print(0, 10);

  double st         = GetTime();
  NVStrings* result = strs->replace_re("\\s+", " ");
  double et         = GetTime() - st;
  printf("result: (%ld bytes)\n", result->memsize());
  result->print(0, 10);
  printf("  %g seconds\n", et);
  NVStrings::destroy(result);
  NVStrings::destroy(strs);
}

void test_normalize()
{
  std::vector<const char*> data_ptrs;
  for (int idx = 0; idx < 100000000; ++idx)
    data_ptrs.push_back(hstrs[idx % hstrs.size()]);  // dist(mt)
  NVStrings* strs = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size());
  printf("strings(%d): (%ld bytes)\n", strs->size(), strs->memsize());
  strs->print(0, 10);

  double st         = GetTime();
  NVStrings* result = NVText::normalize_spaces(*strs);
  double et         = GetTime() - st;
  printf("result: (%ld bytes)\n", result->memsize());
  result->print(0, 10);
  printf("  %g seconds\n", et);
  NVStrings::destroy(result);
  NVStrings::destroy(strs);
}

int main(int argc, char** argv)
{
  test_regex();
  test_normalize();
  return 0;
}
