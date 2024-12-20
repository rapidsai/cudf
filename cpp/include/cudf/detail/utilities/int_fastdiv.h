/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Copyright 2014 Maxim Milakov
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

class int_fastdiv {
 public:
  // divisor != 0
  __host__ __device__ __forceinline__ int_fastdiv(int divisor = 0) : d(divisor)
  {
    update_magic_numbers();
  }

  __host__ __device__ __forceinline__ int_fastdiv& operator=(int divisor)
  {
    this->d = divisor;
    update_magic_numbers();
    return *this;
  }

  __host__ __device__ __forceinline__ operator int() const { return d; }

 private:
  int d;
  int M;
  int s;
  int n_add_sign;

  // Hacker's Delight, Second Edition, Chapter 10, Integer Division By Constants
  __host__ __device__ __forceinline__ void update_magic_numbers()
  {
    if (d == 1) {
      M          = 0;
      s          = -1;
      n_add_sign = 1;
      return;
    } else if (d == -1) {
      M          = 0;
      s          = -1;
      n_add_sign = -1;
      return;
    }

    int p;
    unsigned int ad, anc, delta, q1, r1, q2, r2, t;
    unsigned const two31 = 0x8000'0000u;
    ad                   = (d == 0) ? 1 : abs(d);
    t                    = two31 + ((unsigned int)d >> 31);
    anc                  = t - 1 - t % ad;
    p                    = 31;
    q1                   = two31 / anc;
    r1                   = two31 - q1 * anc;
    q2                   = two31 / ad;
    r2                   = two31 - q2 * ad;
    do {
      ++p;
      q1 = 2 * q1;
      r1 = 2 * r1;
      if (r1 >= anc) {
        ++q1;
        r1 -= anc;
      }
      q2 = 2 * q2;
      r2 = 2 * r2;
      if (r2 >= ad) {
        ++q2;
        r2 -= ad;
      }
      delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    this->M = q2 + 1;
    if (d < 0) this->M = -this->M;
    this->s = p - 32;

    if ((d > 0) && (M < 0))
      n_add_sign = 1;
    else if ((d < 0) && (M > 0))
      n_add_sign = -1;
    else
      n_add_sign = 0;
  }

  __host__ __device__ __forceinline__ friend int operator/(int const divident,
                                                           int_fastdiv const& divisor);
};

__host__ __device__ __forceinline__ int operator/(int const n, int_fastdiv const& divisor)
{
  int q;
#ifdef __CUDA_ARCH__
  asm("mul.hi.s32 %0, %1, %2;" : "=r"(q) : "r"(divisor.M), "r"(n));
#else
  q = (((unsigned long long)((long long)divisor.M * (long long)n)) >> 32);
#endif
  q += n * divisor.n_add_sign;
  if (divisor.s >= 0) {
    q >>= divisor.s;  // we rely on this to be implemented as arithmetic shift
    q += (((unsigned int)q) >> 31);
  }
  return q;
}

__host__ __device__ __forceinline__ int operator%(int const n, int_fastdiv const& divisor)
{
  int quotient  = n / divisor;
  int remainder = n - quotient * divisor;
  return remainder;
}

__host__ __device__ __forceinline__ int operator/(unsigned int const n, int_fastdiv const& divisor)
{
  return ((int)n) / divisor;
}

__host__ __device__ __forceinline__ int operator%(unsigned int const n, int_fastdiv const& divisor)
{
  return ((int)n) % divisor;
}

__host__ __device__ __forceinline__ int operator/(short const n, int_fastdiv const& divisor)
{
  return ((int)n) / divisor;
}

__host__ __device__ __forceinline__ int operator%(short const n, int_fastdiv const& divisor)
{
  return ((int)n) % divisor;
}

__host__ __device__ __forceinline__ int operator/(unsigned short const n,
                                                  int_fastdiv const& divisor)
{
  return ((int)n) / divisor;
}

__host__ __device__ __forceinline__ int operator%(unsigned short const n,
                                                  int_fastdiv const& divisor)
{
  return ((int)n) % divisor;
}

__host__ __device__ __forceinline__ int operator/(char const n, int_fastdiv const& divisor)
{
  return ((int)n) / divisor;
}

__host__ __device__ __forceinline__ int operator%(char const n, int_fastdiv const& divisor)
{
  return ((int)n) % divisor;
}

__host__ __device__ __forceinline__ int operator/(unsigned char const n, int_fastdiv const& divisor)
{
  return ((int)n) / divisor;
}

__host__ __device__ __forceinline__ int operator%(unsigned char const n, int_fastdiv const& divisor)
{
  return ((int)n) % divisor;
}
