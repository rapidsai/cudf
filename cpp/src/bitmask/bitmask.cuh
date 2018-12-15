/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef _BITMASK_H_
#define _BITMASK_H_

#pragma once

#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include <cuda_runtime_api.h>


namespace bitmask {

/* ---------------------------------------------------------------------------- */
/**
 * @Synopsis  Base class for managing bit masks.
 */
/* ---------------------------------------------------------------------------- */
class Util {
public:
  Util(gdf_valid_type *valid, int bitlength): valid_(valid), bitlength_(bitlength) {}
  Util(const Util &util): valid_(util.valid_), bitlength_(util.bitlength_) {}

  /**
   * Check to see if a record is Valid (aka not null)
   *
   * @param[in] record_idx   the record index to check
   *
   * @return  true if record is valid, false if record is null
   */
  template <typename T>
  bool IsValid(T record_idx) const {
    int rec = WhichWord(record_idx);
    int bit = WhichBit(record_idx);

    return ((valid_[rec] & (1U << bit)) != 0);
  }

  /**
   * Set a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  void SetBit(T record_idx) {
    int rec = WhichWord(record_idx);
    int bit = WhichBit(record_idx);

    valid_[rec] = valid_[rec] | (1U << bit);
  }

  /**
   * Clear a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  void ClearBit(T record_idx) {
    int rec = WhichWord(record_idx);
    int bit = WhichBit(record_idx);

    valid_[rec] = valid_[rec] & (~(1U << bit));
  }

  /**
   * Getter for valid
   *
   * @return pointer to valid bitmask
   */
  gdf_valid_type *GetValid() const {
    return valid_;
  }

  /**
   * Get length
   *
   * @return length of bitmask in bits
   */
  int Length() const {
    return bitlength_;
  }

  /**
   * determine the bitmap that contains a record
   * @param[in]  record_idx    The record index
   * @return the bitmap index
   */
  template <typename T>
  int WhichWord(T record_idx) const {
    return (record_idx / GDF_VALID_BITSIZE);
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  int WhichBit(T record_idx) const {
    return (record_idx % GDF_VALID_BITSIZE);
  }

  /**
   * determine how many words need to be allocated
   *
   * @param[in]  number_of_records    The number of bits in the mask
   *
   * @return the number of words to allocate
   */
  template <typename T>
  static int NumWords(T number_of_records) {
    return (number_of_records + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE;
  }

  /**
   * Get the number of words in this bit mask
   *
   * @return the number of words
   */
  inline
  int NumWords() const {
    return NumWords(bitlength_);
  }

private:
  /**
   *   array of entries containing the bitmask
   */
  gdf_valid_type *valid_;
  int             bitlength_;
};


/* ---------------------------------------------------------------------------- */
/**
 * @Synopsis  Class for managing bit masks on the device
 */
/* ---------------------------------------------------------------------------- */
class Device {
public:
   Device(gdf_valid_type *valid, int bitlength_): util_(valid, bitlength_) {}
   Device(const Util &util): util_(util) {}

  /**
   * Check to see if a record is Valid (aka not null)
   *
   * @param[in] record_idx   the record index to check
   *
   * @return  true if record is valid, false if record is null
   */
  template <typename T>
  __host__ bool IsValid(T record_idx) const {
    return util_.IsValid(record_idx);
  }

  /**
   * Set a bit (not thread-save)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void SetBit(T record_idx) {
    util_.SetBit(record_idx);
  }


  /**
   * Clear a bit (not thread-safe)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void ClearBit(T record_idx) {
    util_.ClearBit(record_idx);
  }

  /**
   * Set a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void SetBitThreadsafe(T record_idx) {
    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    atomicOr( &GetValid()[rec], (1U << bit));
  }


  /**
   * Clear a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void ClearBitThreadsafe(T record_idx) {
    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    atomicAnd( &GetValid()[rec], ~(1U << bit));
  }

  /**
   *  Count the number of ones set in the bit mask
   *
   *  @return the number of ones set in the bit mask
   */
  __device__ int CountOnes() const {
    int sum = 0;
    for (int i = 0 ; i < NumWords() ; ++i) {
      sum += __popc(GetValid()[i]);
    }

    return sum;
  }

  /**
   * Get the number of words in this bit mask
   *
   * @return the number of words
   */
  __device__ int NumWords() const {
    return util_.NumWords();
  }

  /**
   * Getter for valid
   *
   * @return pointer to valid bitmask
   */
  __device__ gdf_valid_type *GetValid() const {
    return util_.GetValid();
  }

  /**
   * Get length
   *
   * @return length of bitmask in bits
   */
  int Length() const {
    return util_.Length();
  }

private:
  Util util_;
};

/* ---------------------------------------------------------------------------- */
/**
 * @Synopsis  Class for the host to setup and manipulate bit mask.  Calls
 *            to access bits need to reference memory on the device.
 */
/* ---------------------------------------------------------------------------- */
class Host {
public:
   Host(gdf_valid_type *valid, int bitlength): util_(valid, bitlength) {}
   Host(const Util &util): util_(util) {}

  /**
   * Allocate device space for the valid bitmap.
   *
   * @param[in]  number_of_records     number of records
   * @param[in]  fill_value            optional, should the memory be initialized to all 0 or 1s. All other values indicate un-initialized
   * @return pointer to allocated space on the device
   */
  template <typename T>
  __host__
  static gdf_valid_type * CreateBitmask(T number_of_records, int fill_value = -1) {

    gdf_valid_type *valid_d;

    int num_bitmasks = Util::NumWords(number_of_records);

    RMM_ALLOC((void**)&valid_d, sizeof(gdf_valid_type) * num_bitmasks, 0);

    if (valid_d == NULL)
      return valid_d;

    if (fill_value == 0) {
      cudaMemset(valid_d, 0, sizeof(gdf_valid_type) * num_bitmasks);
    } else if (fill_value == 1) {

      gdf_valid_type temp;
      memset(&temp, 0xff, sizeof(gdf_valid_type));

      for (int i = 0 ; i < (num_bitmasks - 1) ; ++i) {
        PutWord(&temp, valid_d + i);
      }

      //
      //  Need to worry about the case where we set bits past the end
      //
      int bits_used_in_last_word = number_of_records - (num_bitmasks - 1) * GDF_VALID_BITSIZE;

      temp = (temp ^ (temp >> bits_used_in_last_word));

      PutWord(&temp, valid_d + num_bitmasks - 1);
    }

    return valid_d;
  }

  /**
   *  Copy data between host and device
   *
   *  @param[out] dst      - the address of the destination
   *  @param[in] src       - the address of the source
   *  @param[in] num_bits  - the number of bits in the bitmask
   *  @param[in] kind      - the direction of the copy
   */
  inline
  static void CopyBitmask(gdf_valid_type *dst, const gdf_valid_type *src, size_t num_bits, enum cudaMemcpyKind kind) {
    cudaMemcpy(dst, src, Util::NumWords(num_bits) * sizeof(gdf_valid_type), kind);
  }

  /**
   *  Copy data between host and device
   *
   *  @param[out] dst      - the address of the destination
   *  @param[in] src       - the address of the source
   *  @param[in] num_bits  - the number of bits in the bitmask
   *  @param[in] kind      - the direction of the copy
   */
  inline
  static void CopyBitmask(Util &dst, const Util &src, size_t num_bits, enum cudaMemcpyKind kind) {
    CopyBitmask(dst.GetValid(), src.GetValid(), num_bits, kind);
  }

  /**
   * Deallocate device space for the valid bitmap
   *
   * @param[in]  valid   The pointer to device space that we wish to deallocate
   */
  static void DestroyBitmask(gdf_valid_type *valid) {
    RMM_FREE((void **) &valid, 0);
  }

  /**
   * Getter for valid
   *
   * @return pointer to valid bitmask
   */
  gdf_valid_type *GetValid() const {
    return util_.GetValid();
  }

  /**
   * Get length
   *
   * @return length of bitmask in bits
   */
  int Length() const {
    return util_.Length();
  }

  /**
   * Check to see if a record is Valid (aka not null)
   *
   * @param[in] valid        the device memory containing the valid bitmaps
   * @param[in] record_idx   the record index to check
   *
   * @return  true if record is valid, false if record is null
   */
  __host__ bool IsValid(int record_idx) const {

    gdf_valid_type h_bitm;

    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    GetWord(&h_bitm, rec);

    return ((h_bitm & (1U << bit)) != 0);
  }

  /**
   * Set a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __host__ void SetBit(T record_idx) {
    gdf_valid_type h_bitm;

    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    GetWord(&h_bitm, rec);
    h_bitm = h_bitm | (1U << bit);
    PutWord(&h_bitm, rec);
  }

  /**
   * Clear a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __host__ void ClearBit(T record_idx) {
    gdf_valid_type h_bitm;

    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    GetWord(&h_bitm, rec);
    h_bitm = h_bitm & (~(1U << bit));
    PutWord(&h_bitm, rec);
  }

  /**
   * Get the number of words in this bit mask
   *
   * @return the number of words
   */
  __host__ int NumWords() const {
    return util_.NumWords();
  }

private:
  Util util_;

  inline
  static void GetWord(gdf_valid_type *word, const gdf_valid_type *device_word) {
    cudaMemcpy(word, device_word, sizeof(gdf_valid_type), cudaMemcpyDeviceToHost);
  }

  inline
  static void PutWord(const gdf_valid_type *word, gdf_valid_type *device_word) {
    cudaMemcpy(device_word, word, sizeof(gdf_valid_type), cudaMemcpyHostToDevice);
  }

public:
  inline
  void GetWord(gdf_valid_type *word, int rec) const {
    GetWord(word, GetValid()[rec]);
  }

  inline
  void PutWord(gdf_valid_type *word, int rec) {
    PutWord(&GetValid()[rec], word);
  }
};



//----------------------------------------------------------------------------------------------------------
//      Device Mask Utility Functions
//----------------------------------------------------------------------------------------------------------
class HostOnly {
public:
   HostOnly(gdf_valid_type *valid, int bitlength): util_(valid, bitlength) {}
   HostOnly(const Util &util): util_(util) {}

  /**
   * Allocate device space for the valid bitmap.
   *
   * @param[out] gdf_valid_type *      pointer to where device memory will be allocated and returned
   * @param[in]  number_of_records     number of records
   * @param[in]  fill_value            optional, should the memory be initialized to all 0 or 1s. All other values indicate un-initialized
   * @return error status
   */
  template <typename T>
  static gdf_valid_type * CreateBitmask(T number_of_records, int fill_value = -1) {

    gdf_valid_type *valid_d;

    int num_bitmasks = Util::NumWords(number_of_records);

    valid_d = (gdf_valid_type *) malloc(sizeof(gdf_valid_type) * num_bitmasks);

    if (valid_d == NULL)
      return valid_d;

    if (fill_value == 0) {
      memset(valid_d, 0, sizeof(gdf_valid_type) * num_bitmasks);
    } else if (fill_value == 1) {

      if (num_bitmasks > 1) 
        memset(valid_d, 0xff, sizeof(gdf_valid_type) * (num_bitmasks-1));

      //
      //  Need to worry about the case where we set bits past the end
      //
      int bits_used_in_last_word = number_of_records - (num_bitmasks - 1) * GDF_VALID_BITSIZE;

      gdf_valid_type temp = ~0U;
      valid_d[num_bitmasks - 1] = (temp ^ (temp >> bits_used_in_last_word));
    }

    return valid_d;
  }

  /**
   * Deallocate device space for the valid bitmap
   *
   * @param[in]  valid   The pointer to host space that we wish to deallocate
   */
  static void DestroyBitmask(gdf_valid_type *valid) {
    free(valid);
  }

  /**
   * Deallocate device space for the valid bitmap contained in this object.
   *
   * @param[in]  valid   The pointer to host space that we wish to deallocate
   */
  void DestroyBitmask() {
    free(GetValid());
  }

  /**
   * Check to see if a record is Valid (aka not null)
   *
   * @param[in] record_idx   the record index to check
   *
   * @return  true if record is valid, false if record is null
   */
  template <typename T>
  __host__ bool IsValid(T record_idx) const {
    return util_.IsValid(record_idx);
  }

  /**
   * Set a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __host__ void SetBit(T record_idx) {
    util_.SetBit(record_idx);
  }

  /**
   * Clear a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __host__ void ClearBit(T record_idx) {
    util_.ClearBit(record_idx);
  }

  /**
   * Getter for valid
   *
   * @return pointer to valid bitmask
   */
  __host__ gdf_valid_type *GetValid() const {
    return util_.GetValid();
  }

  /**
   * Get length
   *
   * @return length of bitmask in bits
   */
  int Length() const {
    return util_.Length();
  }

  /**
   *  Count the number of ones set in the bit mask
   *
   *  @return the number of ones set in the bit mask
   */
  __host__ int CountOnes() const {
    int sum = 0;
    for (int i = 0 ; i < NumWords() ; ++i) {
      sum += __builtin_popcount(GetValid()[i]);
    }

    return sum;
  }

  /**
   * Get the number of words in this bit mask
   *
   * @return the number of words
   */
  __host__ int NumWords() const {
    return util_.NumWords();
  }

private:
  Util util_;
};

inline HostOnly BitmaskFromDevice(const Host &copy) {
  gdf_valid_type *temp = HostOnly::CreateBitmask(copy.Length(), -1);
  Host::CopyBitmask(temp, copy.GetValid(), copy.Length(), cudaMemcpyDeviceToHost);
  return HostOnly(temp, copy.Length());
}

  


} // end bitmask namespace

#endif
