/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/unique.h>

#include <cudf/utilities/error.hpp>
#include "nvstrings/numeric_category.h"

#define BYTES_FROM_BITS(c) ((c + 7) / 8)

//
__host__ __device__ static bool is_item_null(const BYTE* nulls, int idx)
{
  return nulls && ((nulls[idx / 8] & (1 << (idx % 8))) == 0);
}

size_t count_nulls(const BYTE* nulls, size_t count);

//
template <typename T>
class numeric_category_impl {
 public:
  rmm::device_vector<T> _keys;
  rmm::device_vector<int> _values;
  rmm::device_vector<BYTE> _bitmask;
  bool bkeyset_includes_null{false};

  void init_keys(const T* items, size_t count)
  {
    _keys.resize(count);
    auto execpol = rmm::exec_policy(0);
    thrust::copy(execpol->on(0), items, items + count, _keys.data().get());
  }

  void init_keys(const T* items, const int* indexes, size_t count)
  {
    _keys.resize(count);
    auto execpol = rmm::exec_policy(0);
    thrust::gather(execpol->on(0), indexes, indexes + count, items, _keys.data().get());
  }

  const T* get_keys() { return _keys.data().get(); }

  size_t keys_count() { return _keys.size(); }

  int* get_values(size_t count)
  {
    _values.resize(count);
    return _values.data().get();
  }

  const int* get_values() { return _values.data().get(); }

  size_t values_count() { return _values.size(); }

  void set_values(const int* vals, size_t count)
  {
    _values.resize(count);
    auto execpol = rmm::exec_policy(0);
    thrust::copy(execpol->on(0), vals, vals + count, _values.data().get());
  }

  BYTE* get_nulls(size_t count)
  {
    if (count == 0) return nullptr;
    auto execpol      = rmm::exec_policy(0);
    size_t byte_count = (count + 7) / 8;
    _bitmask.resize(byte_count);
    thrust::fill(execpol->on(0), _bitmask.begin(), _bitmask.end(), 0);
    return _bitmask.data().get();
  }

  const BYTE* get_nulls()
  {
    if (_bitmask.empty()) return nullptr;
    return _bitmask.data().get();
  }

  void set_nulls(const BYTE* nulls, size_t count)
  {
    if (nulls == nullptr || count == 0) return;
    size_t byte_count = (count + 7) / 8;
    _bitmask.resize(byte_count);
    CUDA_TRY(cudaMemcpyAsync(_bitmask.data().get(), nulls, byte_count, cudaMemcpyDeviceToDevice));
    bkeyset_includes_null = count_nulls(nulls, count) > 0;
  }

  void reset_nulls()
  {
    _bitmask.resize(0);
    bkeyset_includes_null = false;
  }
};

template <typename T>
numeric_category<T>::numeric_category()
{
  pImpl = new numeric_category_impl<T>;
}

template <typename T>
numeric_category<T>::numeric_category(const numeric_category&) : pImpl(nullptr)
{
}

template <typename T>
numeric_category<T>::~numeric_category()
{
  delete pImpl;
}

template <typename T>
struct sort_functor {
  const T* items;
  const BYTE* nulls;
  __device__ bool operator()(int lhs, int rhs)
  {
    bool lhs_null = is_item_null(nulls, lhs);
    bool rhs_null = is_item_null(nulls, rhs);
    if (lhs_null || rhs_null) return !rhs_null;  // sorts: null < non-null
    return items[lhs] < items[rhs];
  }
};

template <typename T>
struct copy_unique_functor {
  const T* items;
  const BYTE* nulls;
  int* indexes;
  int* values;
  __device__ bool operator()(int idx)
  {
    if (idx == 0) {
      values[0] = 0;
      return true;
    }
    int lhs = indexes[idx - 1], rhs = indexes[idx];
    // printf("%d:%d,%d\n",idx,lhs,rhs);
    bool lhs_null = is_item_null(nulls, lhs), rhs_null = is_item_null(nulls, rhs);
    bool isunique = true;
    if (lhs_null || rhs_null)
      isunique = (lhs_null != rhs_null);
    else
      isunique = (items[lhs] != items[rhs]);
    values[idx] = (int)isunique;
    return isunique;
  }
};

template <typename T>
numeric_category<T>::numeric_category(const T* items, size_t count, const BYTE* nulls)
  : pImpl(nullptr)
{
  pImpl = new numeric_category_impl<T>;
  if (!items || !count) return;  // empty numeric_category
  auto execpol = rmm::exec_policy(0);
  rmm::device_vector<int> indexes(count);
  thrust::sequence(execpol->on(0), indexes.begin(), indexes.end());  // 0,1,2,3,4,5,6,7,8
  thrust::sort(execpol->on(0), indexes.begin(), indexes.end(), sort_functor<T>{items, nulls});
  int* d_values  = pImpl->get_values(count);
  int* d_indexes = indexes.data().get();
  rmm::device_vector<int> map_indexes(count);
  int* d_map_indexes = map_indexes.data().get();
  int* d_map_nend    = thrust::copy_if(execpol->on(0),
                                    thrust::make_counting_iterator<int>(0),
                                    thrust::make_counting_iterator<int>(count),
                                    d_map_indexes,
                                    copy_unique_functor<T>{items, nulls, d_indexes, d_values});
  int ucount         = (int)(d_map_nend - d_map_indexes);
  rmm::device_vector<int> keys_indexes(ucount);  // get the final indexes to the unique keys
  thrust::gather(execpol->on(0), d_map_indexes, d_map_nend, indexes.begin(), keys_indexes.begin());
  // scan will produce the resulting values
  thrust::inclusive_scan(execpol->on(0), d_values, d_values + count, d_values);
  // sort will put them in the correct order
  thrust::sort_by_key(execpol->on(0), indexes.begin(), indexes.end(), d_values);
  // gather the keys for this numeric_category
  pImpl->init_keys(items, keys_indexes.data().get(), ucount);
  // just make a copy of the nulls bitmask
  if (nulls) pImpl->set_nulls(nulls, count);
}

template <typename T>
size_t numeric_category<T>::size()
{
  return pImpl->values_count();
}

template <typename T>
size_t numeric_category<T>::keys_size()
{
  return pImpl->keys_count();
}

template <typename T>
const T* numeric_category<T>::keys()
{
  return pImpl->get_keys();
}

template <typename T>
const int* numeric_category<T>::values()
{
  return pImpl->get_values();
}

template <typename T>
const T numeric_category<T>::get_key_for(int idx)
{
  return pImpl->_keys[idx];
}  //

template <typename T>
const BYTE* numeric_category<T>::nulls_bitmask()
{
  return pImpl->get_nulls();
}

template <typename T>
bool numeric_category<T>::has_nulls()
{
  return count_nulls(pImpl->get_nulls(), pImpl->values_count()) > 0;
}

template <typename T>
bool numeric_category<T>::keys_have_null()
{
  return pImpl->bkeyset_includes_null;
}

template <typename T>
void numeric_category<T>::print(const char* prefix, const char* delimiter)
{
  std::cout << prefix;
  size_t count = pImpl->keys_count();
  if (count == 0) std::cout << "<no keys>";
  const T* d_keys = pImpl->get_keys();
  thrust::host_vector<T> h_keys(count);
  CUDA_TRY(cudaMemcpyAsync(h_keys.data(), d_keys, count * sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t idx = 0; idx < count; ++idx) {
    if (idx || !pImpl->bkeyset_includes_null) {
      if (std::is_same<T, char>::value)
        std::cout << (int)h_keys[idx];  // want int8
      else
        std::cout << h_keys[idx];
    } else
      std::cout << "-";
    std::cout << delimiter;
  }
  std::cout << "\n";

  std::cout << prefix;
  count = pImpl->values_count();
  if (count == 0) std::cout << "<no values>";
  const int* d_values = pImpl->get_values();
  thrust::host_vector<int> h_values(count);
  CUDA_TRY(cudaMemcpyAsync(h_values.data(), d_values, count * sizeof(int), cudaMemcpyDeviceToHost));
  const BYTE* d_nulls = pImpl->get_nulls();
  size_t byte_count   = (count + 7) / 8;
  thrust::host_vector<BYTE> nulls(byte_count);
  BYTE* h_nulls = nullptr;
  if (d_nulls) {
    h_nulls = nulls.data();
    CUDA_TRY(cudaMemcpy(h_nulls, d_nulls, byte_count * sizeof(BYTE), cudaMemcpyDeviceToHost));
  }
  for (size_t idx = 0; idx < count; ++idx) {
    if (is_item_null(h_nulls, idx))
      std::cout << "-";
    else
      std::cout << h_values[idx];
    std::cout << delimiter;
  }
  std::cout << "\n";
}

template <typename T>
numeric_category<T>* numeric_category<T>::copy()
{
  numeric_category<T>* result = new numeric_category<T>;
  result->pImpl->init_keys(pImpl->get_keys(), pImpl->keys_count());
  result->pImpl->set_values(pImpl->get_values(), pImpl->values_count());
  result->pImpl->set_nulls(pImpl->get_nulls(), pImpl->values_count());
  return result;
}

template <typename T>
int numeric_category<T>::get_index_for(T key)
{
  auto execpol        = rmm::exec_policy(0);
  const int* d_values = pImpl->get_values();
  int index = -1, count = keys_size();
  const T* d_keys = pImpl->get_keys();
  rmm::device_vector<int> d_index(1);
  thrust::copy_if(execpol->on(0),
                  thrust::make_counting_iterator<int>(0),
                  thrust::make_counting_iterator<int>(count),
                  d_index.begin(),
                  [d_keys, key] __device__(int idx) { return key == d_keys[idx]; });
  index = d_index[0];
  return index;
}

struct indexes_for_fn {
  const int* d_values;
  int index;
  __device__ bool operator()(int idx) { return d_values[idx] == index; }
};

template <typename T>
size_t numeric_category<T>::get_indexes_for(T key, int* d_results)
{
  auto execpol        = rmm::exec_policy(0);
  int index           = get_index_for(key);
  const int* d_values = pImpl->get_values();
  size_t count        = size();
  if (d_results == nullptr) return thrust::count(execpol->on(0), d_values, d_values + count, index);
  int* nend = thrust::copy_if(execpol->on(0),
                              thrust::make_counting_iterator<int>(0),
                              thrust::make_counting_iterator<int>(count),
                              d_results,
                              indexes_for_fn{d_values, index});
  return (size_t)(nend - d_results);
}

template <typename T>
size_t numeric_category<T>::get_indexes_for_null_key(int* d_results)
{
  if (pImpl->get_nulls() == nullptr) return 0;  // there are no null entries
  auto execpol        = rmm::exec_policy(0);
  int index           = 0;  // null key is always index 0
  const int* d_values = pImpl->get_values();
  size_t count        = thrust::count(execpol->on(0), d_values, d_values + size(), index);
  if (d_results == nullptr) return count;
  thrust::copy_if(execpol->on(0),
                  thrust::make_counting_iterator<int>(0),
                  thrust::make_counting_iterator<int>(count),
                  d_results,
                  indexes_for_fn{d_values, index});
  return count;
}

template <typename T>
void numeric_category<T>::to_type(T* d_results, BYTE* nulls)
{
  auto execpol        = rmm::exec_policy(0);
  const int* d_values = pImpl->get_values();
  size_t count        = pImpl->values_count();
  thrust::gather(execpol->on(0), d_values, d_values + count, pImpl->get_keys(), d_results);
  const BYTE* d_nulls = pImpl->get_nulls();
  if (d_nulls && nulls)
    CUDA_TRY(cudaMemcpyAsync(nulls, d_nulls, ((count + 7) / 8), cudaMemcpyDeviceToDevice));
}

template <typename T>
void numeric_category<T>::gather_type(const int* d_indexes, size_t count, T* d_results, BYTE* nulls)
{
  auto execpol = rmm::exec_policy(0);
  // should these be indexes of the values and not values themselves?
  size_t kcount = keys_size();
  int check =
    thrust::count_if(execpol->on(0), d_indexes, d_indexes + count, [kcount] __device__(int val) {
      return (val < 0) || (val >= kcount);
    });
  if (check > 0) throw std::out_of_range("gather_type invalid index value");
  thrust::gather(execpol->on(0), d_indexes, d_indexes + count, pImpl->get_keys(), d_results);
  // need to also gather the null bits
  const BYTE* d_nulls = pImpl->get_nulls();
  if (nulls) {
    bool include_null = pImpl->bkeyset_includes_null;
    size_t byte_count = (count + 7) / 8;
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<size_t>(0),
                       byte_count,
                       [d_indexes, count, include_null, nulls] __device__(size_t byte_idx) {
                         BYTE mask = 0;
                         for (int bit = 0; bit < 8; ++bit) {
                           size_t idx = byte_idx * 8 + bit;
                           if (idx < count)
                             mask |= (int)!(include_null && (d_indexes[idx] == 0)) << bit;
                         }
                         nulls[byte_idx] = mask;
                       });
  }
}

template <typename T>
struct sort_update_keys_fn {
  const T* keys;
  size_t kcount;
  bool include_null;
  const BYTE* nulls;
  __device__ bool operator()(int lhs, int rhs)
  {
    bool lhs_null =
      ((lhs == 0) && include_null) || ((lhs >= kcount) && is_item_null(nulls, lhs - kcount));
    bool rhs_null =
      ((rhs == 0) && include_null) || ((rhs >= kcount) && is_item_null(nulls, rhs - kcount));
    if (lhs_null || rhs_null) return !rhs_null;  // sorts: null < non-null
    return keys[lhs] < keys[rhs];
  }
};

template <typename T>
struct unique_update_keys_fn {
  const T* keys;
  size_t kcount;
  bool include_null;
  const BYTE* nulls;
  __device__ bool operator()(int lhs, int rhs)
  {
    bool lhs_null =
      ((lhs == 0) && include_null) || ((lhs >= kcount) && is_item_null(nulls, lhs - kcount));
    bool rhs_null =
      ((rhs == 0) && include_null) || ((rhs >= kcount) && is_item_null(nulls, rhs - kcount));
    if (lhs_null || rhs_null) return lhs_null == rhs_null;
    return keys[lhs] == keys[rhs];
  }
};

// this is almost a straight gather; maybe gather_if?
struct remap_values_fn {
  const int* values;
  const int* map;
  int* new_values;
  __device__ void operator()(size_t idx)
  {
    int value       = values[idx];
    new_values[idx] = (value < 0 ? value : map[value]);
  }
};

template <typename T>
numeric_category<T>* numeric_category<T>::add_keys(const T* items, size_t count, const BYTE* nulls)
{
  if (!items || !count) return copy();
  auto execpol                = rmm::exec_policy(0);
  numeric_category<T>* result = new numeric_category<T>;
  size_t kcount               = keys_size();
  // incorporating the keys and adjust the values
  bool include_null = pImpl->bkeyset_includes_null;
  const T* d_keys   = pImpl->get_keys();
  size_t both_count = kcount + count;           // this is not the unique count
  rmm::device_vector<T> both_keys(both_count);  // first combine both keysets
  T* d_both_keys = both_keys.data().get();
  thrust::copy(execpol->on(0), d_keys, d_keys + kcount, d_both_keys);
  thrust::copy(execpol->on(0), items, items + count, d_both_keys + kcount);
  rmm::device_vector<int> xvals(both_count);
  int* d_xvals = xvals.data().get();  // build vector like: 0,...,(kcount-1),-1,...,-count
  thrust::tabulate(execpol->on(0), d_xvals, d_xvals + both_count, [kcount] __device__(int idx) {
    return (idx < kcount) ? idx : (kcount - idx - 1);
  });
  // compute the new keyset by doing sort/unique
  rmm::device_vector<int> indexes(both_count);
  thrust::sequence(execpol->on(0), indexes.begin(), indexes.end());
  int* d_indexes = indexes.data().get();
  // stable-sort preserves order for keys that match
  thrust::stable_sort_by_key(execpol->on(0),
                             d_indexes,
                             d_indexes + both_count,
                             d_xvals,
                             sort_update_keys_fn<T>{d_both_keys, kcount, include_null, nulls});
  auto nend =
    thrust::unique_by_key(execpol->on(0),
                          d_indexes,
                          d_indexes + both_count,
                          d_xvals,
                          unique_update_keys_fn<T>{d_both_keys, kcount, include_null, nulls});
  size_t unique_count = nend.second - d_xvals;
  result->pImpl->init_keys(d_both_keys, d_indexes, unique_count);
  // done with keys
  // update the values to their new positions using the xvals created above
  if (size()) {
    size_t vcount       = size();
    const int* d_values = pImpl->get_values();
    int* d_new_values   = result->pImpl->get_values(vcount);
    // map the new positions
    rmm::device_vector<int> yvals(kcount, -1);
    int* d_yvals = yvals.data().get();
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<size_t>(0),
                       unique_count,
                       [d_yvals, d_xvals] __device__(size_t idx) {
                         int map_id = d_xvals[idx];
                         if (map_id >= 0) d_yvals[map_id] = idx;
                       });
    // apply new positions to new numeric_category values
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<size_t>(0),
                       vcount,
                       remap_values_fn{d_values, d_yvals, d_new_values});
  }
  // handle nulls
  result->pImpl->set_nulls(pImpl->get_nulls(), size());
  if (!include_null)                                                       // check if a null
    result->pImpl->bkeyset_includes_null = count_nulls(nulls, count) > 0;  // key was added
  return result;
}

template <typename T>
numeric_category<T>* numeric_category<T>::remove_keys(const T* items,
                                                      size_t count,
                                                      const BYTE* nulls)
{
  if (!items || !count) return copy();
  auto execpol                = rmm::exec_policy(0);
  numeric_category<T>* result = new numeric_category<T>;
  size_t kcount               = keys_size();
  size_t both_count           = kcount + count;
  const T* d_keys             = pImpl->get_keys();
  rmm::device_vector<T> both_keys(both_count);  // first combine both keysets
  T* d_both_keys = both_keys.data().get();
  thrust::copy(execpol->on(0), d_keys, d_keys + kcount, d_both_keys);        // these keys
  thrust::copy(execpol->on(0), items, items + count, d_both_keys + kcount);  // and those keys
  rmm::device_vector<int> xvals(both_count);
  int* d_xvals = xvals.data().get();  // build vector like: 0,...,(kcount-1),-1,...,-count
  thrust::tabulate(execpol->on(0), d_xvals, d_xvals + both_count, [kcount] __device__(int idx) {
    return (idx < kcount) ? idx : (kcount - idx - 1);
  });
  // compute the new keyset by doing sort/unique
  rmm::device_vector<int> indexes(both_count);
  thrust::sequence(execpol->on(0), indexes.begin(), indexes.end());
  int* d_indexes    = indexes.data().get();
  bool include_null = pImpl->bkeyset_includes_null;
  // stable-sort preserves order for keys that match
  thrust::stable_sort_by_key(execpol->on(0),
                             d_indexes,
                             d_indexes + both_count,
                             d_xvals,
                             sort_update_keys_fn<T>{d_both_keys, kcount, include_null, nulls});
  size_t unique_count = both_count;
  {
    rmm::device_vector<int> map_indexes(both_count);
    int* d_map_indexes = map_indexes.data().get();
    int* d_end         = thrust::copy_if(
      execpol->on(0),
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(both_count),
      d_map_indexes,
      [d_both_keys, kcount, both_count, d_indexes, d_xvals, include_null, nulls] __device__(
        int idx) {
        if (d_xvals[idx] < 0) return false;
        if (idx == both_count - 1) return true;
        int lhs = d_indexes[idx], rhs = d_indexes[idx + 1];
        bool lhs_null =
          ((lhs == 0) && include_null) || ((lhs >= kcount) && is_item_null(nulls, lhs - kcount));
        bool rhs_null =
          ((rhs == 0) && include_null) || ((rhs >= kcount) && is_item_null(nulls, rhs - kcount));
        if (lhs_null || rhs_null) return lhs_null != rhs_null;
        return (d_both_keys[lhs] != d_both_keys[rhs]);
      });
    unique_count = (size_t)(d_end - d_map_indexes);
    rmm::device_vector<int> keys_indexes(unique_count);
    thrust::gather(execpol->on(0), d_map_indexes, d_end, d_indexes, keys_indexes.data().get());
    result->pImpl->init_keys(d_both_keys, keys_indexes.data().get(), unique_count);
    // setup for the value remap
    rmm::device_vector<int> new_xvals(unique_count);
    thrust::gather(execpol->on(0), d_map_indexes, d_end, d_xvals, new_xvals.data().get());
    xvals.swap(new_xvals);
    d_xvals = xvals.data().get();
  }
  // done with the keys
  // now remap values to their new positions
  size_t vcount = size();
  if (vcount) {
    const int* d_values = values();
    int* d_new_values   = result->pImpl->get_values(vcount);
    // values pointed to removed keys will now have index=-1
    rmm::device_vector<int> yvals(kcount, -1);
    int* d_yvals = yvals.data().get();
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<int>(0),
                       (int)unique_count,
                       [d_yvals, d_xvals] __device__(int idx) { d_yvals[d_xvals[idx]] = idx; });
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<int>(0),
                       (int)vcount,
                       remap_values_fn{d_values, d_yvals, d_new_values});
  }
  // finally, handle the nulls
  // if null key is removed, then null values are abandoned (become -1)
  const BYTE* d_nulls = pImpl->get_nulls();
  if (d_nulls) {
    if (count_nulls(nulls, count))  // removed null key; values should be -1 anyways
      result->pImpl->reset_nulls();
    else  // otherwise just copy them
      result->pImpl->set_nulls(d_nulls, vcount);
  }
  return result;
}

template <typename T>
numeric_category<T>* numeric_category<T>::remove_unused_keys()
{
  size_t kcount = pImpl->keys_count();
  if (kcount == 0) return copy();
  auto execpol        = rmm::exec_policy(0);
  const int* d_values = pImpl->get_values();
  rmm::device_vector<int> usedkeys(kcount, 0);
  int* d_usedkeys = usedkeys.data().get();
  // find the keys that not being used
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<int>(0),
                     (int)size(),
                     [d_values, d_usedkeys] __device__(int idx) {
                       int pos = d_values[idx];
                       if (pos >= 0) d_usedkeys[pos] = 1;  //
                     });
  // compute how many are not used
  size_t count = kcount - thrust::reduce(execpol->on(0), d_usedkeys, d_usedkeys + kcount, (int)0);
  if (count == 0) return copy();
  //
  rmm::device_vector<T> rmv_keys(count);
  T* d_rmv_keys = rmv_keys.data().get();
  rmm::device_vector<int> indexes(count);
  int* d_indexes = indexes.data().get();
  thrust::copy_if(execpol->on(0),
                  thrust::make_counting_iterator<int>(0),
                  thrust::make_counting_iterator<int>(kcount),
                  d_indexes,
                  [d_usedkeys] __device__(int idx) { return (d_usedkeys[idx] == 0); });
  thrust::gather(execpol->on(0), d_indexes, d_indexes + count, pImpl->get_keys(), d_rmv_keys);
  // handle null key case
  size_t mask_bytes = (count + 7) / 8;
  rmm::device_vector<BYTE> null_keys(mask_bytes, 0);
  BYTE* d_null_keys = nullptr;
  if (pImpl->bkeyset_includes_null && (usedkeys[0] == 0)) {
    d_null_keys = null_keys.data().get();
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<size_t>(0),
                       mask_bytes,
                       [count, d_null_keys] __device__(size_t idx) {
                         size_t base_idx = idx * 8;
                         BYTE mask       = 0xFF;
                         if ((base_idx + 8) > count) mask = mask >> (base_idx + 8 - count);
                         if (idx == 0) mask = mask & 0xFE;
                         d_null_keys[base_idx] = mask;
                       });
  }
  return remove_keys(d_rmv_keys, count, d_null_keys);
}

template <typename T>
numeric_category<T>* numeric_category<T>::set_keys(const T* items, size_t count, const BYTE* nulls)
{
  auto execpol      = rmm::exec_policy(0);
  size_t kcount     = keys_size();
  size_t both_count = kcount + count;  // this is not the unique count
  const T* d_keys   = pImpl->get_keys();
  rmm::device_vector<T> both_keys(both_count);  // first combine both keysets
  T* d_both_keys = both_keys.data().get();
  thrust::copy(execpol->on(0), d_keys, d_keys + kcount, d_both_keys);        // these keys
  thrust::copy(execpol->on(0), items, items + count, d_both_keys + kcount);  // and those keys
  rmm::device_vector<int> xvals(both_count);  // seq-vector for resolving old/new keys
  int* d_xvals = xvals.data().get();          // build vector like: 0,...,(kcount-1),-1,...,-count
  thrust::tabulate(execpol->on(0), d_xvals, d_xvals + both_count, [kcount] __device__(int idx) {
    return (idx < kcount) ? idx : (kcount - idx - 1);
  });
  // sort the combined keysets
  rmm::device_vector<int> indexes(both_count);
  thrust::sequence(execpol->on(0), indexes.begin(), indexes.end());
  int* d_indexes    = indexes.data().get();
  bool include_null = pImpl->bkeyset_includes_null;
  // stable-sort preserves order for keys that match
  thrust::stable_sort_by_key(execpol->on(0),
                             d_indexes,
                             d_indexes + both_count,
                             d_xvals,
                             sort_update_keys_fn<T>{d_both_keys, kcount, include_null, nulls});
  rmm::device_vector<int> map_indexes(both_count);  // needed for gather methods
  int* d_map_indexes = map_indexes.data().get();    // indexes of keys from key1 not in key2
  int* d_copy_end    = thrust::copy_if(
    execpol->on(0),
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(both_count),
    d_map_indexes,
    [d_both_keys, kcount, both_count, d_indexes, d_xvals, include_null, nulls] __device__(int idx) {
      if (d_xvals[idx] < 0) return true;
      if (idx == (both_count - 1)) return false;
      int lhs = d_indexes[idx], rhs = d_indexes[idx + 1];
      bool lhs_null =
        ((lhs == 0) && include_null) || ((lhs >= kcount) && is_item_null(nulls, lhs - kcount));
      bool rhs_null =
        ((rhs == 0) && include_null) || ((rhs >= kcount) && is_item_null(nulls, rhs - kcount));
      if (lhs_null || rhs_null) return lhs_null == rhs_null;
      return (d_both_keys[lhs] == d_both_keys[rhs]);
    });
  size_t copy_count = d_copy_end - d_map_indexes;
  if (copy_count < both_count) {  // if keys are removed, we need new keyset; the gather()s here
                                  // will select the remaining keys
    rmm::device_vector<int> copy_indexes(copy_count);
    rmm::device_vector<int> copy_xvals(copy_count);
    thrust::gather(execpol->on(0),
                   d_map_indexes,
                   d_map_indexes + copy_count,
                   d_indexes,
                   copy_indexes.data().get());  // likely, these 2 lines can be
    thrust::gather(execpol->on(0),
                   d_map_indexes,
                   d_map_indexes + copy_count,
                   d_xvals,
                   copy_xvals.data().get());  // combined with a zip-iterator
    indexes.swap(copy_indexes);
    xvals.swap(copy_xvals);
    d_indexes  = indexes.data().get();
    d_xvals    = xvals.data().get();
    both_count = copy_count;
  }
  // resolve final key-set
  auto d_unique_end =
    thrust::unique_by_key(execpol->on(0),
                          d_indexes,
                          d_indexes + both_count,
                          d_xvals,
                          unique_update_keys_fn<T>{d_both_keys, kcount, include_null, nulls});
  size_t unique_count         = d_unique_end.second - d_xvals;  // both_count - matched;
  numeric_category<T>* result = new numeric_category<T>;
  result->pImpl->init_keys(d_both_keys, d_indexes, unique_count);
  // done with keys, remap the values
  size_t vcount = size();
  if (vcount) {
    const int* d_values = values();
    rmm::device_vector<int> yvals(kcount, -1);  // create map/stencil from old key positions
    int* d_yvals = yvals.data().get();
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<int>(0),
                       (int)unique_count,
                       [d_xvals, d_yvals] __device__(int idx) {
                         int value = d_xvals[idx];
                         if (value >= 0) d_yvals[value] = idx;  // map to new position
                       });
    // create new values using the map in yvals
    int* d_new_values = result->pImpl->get_values(vcount);
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<int>(0),
                       (int)vcount,
                       remap_values_fn{d_values, d_yvals, d_new_values});
    // handles nulls
    bool include_new_null = (count_nulls(nulls, count) > 0);
    if (include_null == include_new_null)
      result->pImpl->set_nulls(pImpl->get_nulls(), vcount);  // nulls do not change position
    else if (include_null)
      result->pImpl->reset_nulls();  // null has been removed
    else
      result->pImpl->bkeyset_includes_null = include_new_null;  // null has been added
  }
  return result;
}

template <typename T>
numeric_category<T>* numeric_category<T>::merge(numeric_category<T>& cat)
{
  auto execpol = rmm::exec_policy(0);
  // first, copy keys so we can sort/unique
  size_t kcount        = pImpl->keys_count();
  size_t count         = kcount + cat.pImpl->keys_count();
  const T* d_keys      = pImpl->get_keys();
  const T* d_catkeys   = cat.pImpl->get_keys();
  bool include_null    = pImpl->bkeyset_includes_null;
  bool catinclude_null = cat.pImpl->bkeyset_includes_null;

  rmm::device_vector<T> keyset(count);
  T* d_keyset = keyset.data().get();
  thrust::copy(execpol->on(0), d_keys, d_keys + kcount, d_keyset);
  thrust::copy(execpol->on(0), d_catkeys, d_catkeys + cat.keys_size(), d_keyset + kcount);
  // build sequence vector and sort positions
  rmm::device_vector<int> indexes(count), xvals(count);
  thrust::sequence(execpol->on(0), indexes.begin(), indexes.end());
  thrust::sequence(execpol->on(0), xvals.begin(), xvals.end());
  int* d_indexes = indexes.data().get();
  int* d_xvals   = xvals.data().get();
  // stable-sort preserves order
  thrust::stable_sort_by_key(
    execpol->on(0),
    d_indexes,
    d_indexes + count,
    d_xvals,
    [d_keyset, kcount, include_null, catinclude_null] __device__(int lhs, int rhs) {
      bool lhs_null = ((lhs == 0) && include_null) || ((lhs == kcount) && catinclude_null);
      bool rhs_null = ((rhs == 0) && include_null) || ((rhs == kcount) && catinclude_null);
      if (lhs_null || rhs_null) return !rhs_null;
      return d_keyset[lhs] < d_keyset[rhs];
    });

  // build anti-matching indicator vector and unique-map at the same time
  rmm::device_vector<int> map_indexes(count), yvals(count);
  int* d_map_indexes = map_indexes.data().get();  // this will contain map to unique indexes
  int* d_yvals =
    yvals.data().get();  // this will have 1's where adjacent keys are different (anti-matching)
  int* d_map_nend = thrust::copy_if(
    execpol->on(0),
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(count),
    d_map_indexes,
    [d_keyset, d_indexes, kcount, d_yvals, include_null, catinclude_null] __device__(int idx) {
      if (idx == 0) {
        d_yvals[0] = 0;
        return true;
      }
      int lhs = d_indexes[idx - 1], rhs = d_indexes[idx];
      bool lhs_null = ((lhs == 0) && include_null) || ((lhs == kcount) && catinclude_null);
      bool rhs_null = ((rhs == 0) && include_null) || ((rhs == kcount) && catinclude_null);
      bool isunique = true;
      if (lhs_null || rhs_null)
        isunique = (lhs_null != rhs_null);
      else
        isunique = (d_keyset[lhs] != d_keyset[rhs]);
      d_yvals[idx] = (int)isunique;
      return isunique;
    });
  size_t unique_count = (size_t)(d_map_nend - d_map_indexes);
  // build the unique indexes by using the map on the sorted indexes
  rmm::device_vector<int> keys_indexes(unique_count);
  thrust::gather(execpol->on(0), d_map_indexes, d_map_nend, indexes.begin(), keys_indexes.begin());
  // build the result
  numeric_category<T>* result = new numeric_category<T>;
  result->pImpl->init_keys(d_keyset, keys_indexes.data().get(), unique_count);
  // done with keys
  // create index to map old positions to their new indexes
  thrust::inclusive_scan(execpol->on(0), d_yvals, d_yvals + count, d_yvals);  // new positions
  thrust::sort_by_key(
    execpol->on(0), d_xvals, d_xvals + count, d_yvals);  // creates map from old to new positions
  int* d_new_values = result->pImpl->get_values(size() + cat.size());  // alloc output values
  // the remap is done in sections and could be combined into a single kernel with branching
  if (size()) {  // remap our values first
    const int* d_values = values();
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<int>(0),
                       (int)size(),
                       remap_values_fn{d_values, d_yvals, d_new_values});
    d_new_values += size();  // point to the
    d_yvals += keys_size();  // next section
  }
  if (cat.size()) {  // remap arg's values
    const int* d_values = cat.values();
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<int>(0),
                       (int)cat.size(),
                       remap_values_fn{d_values, d_yvals, d_new_values});
  }
  // the nulls are just appended together
  const BYTE* d_nulls    = pImpl->get_nulls();
  const BYTE* d_catnulls = cat.pImpl->get_nulls();
  if (d_nulls || d_catnulls) {
    size_t vcount        = size();
    size_t ncount        = vcount + cat.size();
    BYTE* d_result_nulls = result->pImpl->get_nulls(ncount);
    size_t byte_count    = (ncount + 7) / 8;
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<size_t>(0),
      byte_count,
      [d_nulls, d_catnulls, vcount, ncount, d_result_nulls] __device__(size_t byte_idx) {
        BYTE mask = 0;
        for (int bit = 0; bit < 8; ++bit) {
          size_t idx = (byte_idx * 8) + bit;
          if (idx >= ncount) break;
          int flag;
          if (idx < vcount)
            flag = (int)!is_item_null(d_nulls, idx);
          else
            flag = (int)!is_item_null(d_catnulls, (idx - vcount));
          mask |= flag << bit;
        }
        d_result_nulls[byte_idx] = mask;
      });
  }
  result->pImpl->bkeyset_includes_null = include_null || catinclude_null;
  return result;
}

// only valid for null included in keyset
struct gather_nullbits_fn {
  const int* indexes;
  size_t count;
  BYTE* d_nulls;
  __device__ void operator()(size_t byte_idx)
  {
    BYTE mask = 0;
    for (int bit = 0; bit < 8; ++bit) {
      size_t idx = (byte_idx * 8) + bit;
      if (idx < count) mask |= (int)(indexes[idx] != 0) << bit;
    }
    d_nulls[byte_idx] = mask;
  }
};

template <typename T>
numeric_category<T>* numeric_category<T>::gather_and_remap(const int* indexes, size_t count)
{
  auto execpol  = rmm::exec_policy(0);
  size_t kcount = keys_size();
  int check =
    thrust::count_if(execpol->on(0), indexes, indexes + count, [kcount] __device__(int val) {
      return (val < 0) || (val >= kcount);
    });
  if (check > 0) throw std::out_of_range("gather: invalid index value");
  // create histogram-ish record of keys for this gather
  rmm::device_vector<int> xvals(kcount, 0);
  int* d_xvals = xvals.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<size_t>(0),
                     count,
                     [d_xvals, indexes] __device__(size_t idx) { d_xvals[indexes[idx]] = 1; });
  // create indexes of our keys for the new numeric_category
  rmm::device_vector<int> yvals(kcount, 0);
  int* d_yvals        = yvals.data().get();
  auto d_new_end      = thrust::copy_if(execpol->on(0),
                                   thrust::make_counting_iterator<int>(0),
                                   thrust::make_counting_iterator<int>(kcount),
                                   d_yvals,
                                   [d_xvals] __device__(int idx) { return d_xvals[idx] == 1; });
  size_t unique_count = (size_t)(d_new_end - d_yvals);
  // create new numeric_category and set the keys
  numeric_category<T>* result = new numeric_category<T>;
  result->pImpl->init_keys(pImpl->get_keys(), d_yvals, unique_count);
  // now create values by mapping our values over to the new key positions
  thrust::exclusive_scan(
    execpol->on(0), d_xvals, d_xvals + kcount, d_yvals);  // reuse yvals for the map
  int* d_new_values = result->pImpl->get_values(count);
  thrust::gather(execpol->on(0), indexes, indexes + count, d_yvals, d_new_values);
  // also need to gather nulls
  if (pImpl->bkeyset_includes_null && indexes) {
    BYTE* d_new_nulls = result->pImpl->get_nulls(count);
    size_t byte_count = (count + 7) / 8;
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<size_t>(0),
                       byte_count,
                       gather_nullbits_fn{indexes, count, d_new_nulls});
    result->pImpl->bkeyset_includes_null = (count_nulls(d_new_nulls, count) > 0);
  }
  return result;
}

template <typename T>
numeric_category<T>* numeric_category<T>::gather(const int* indexes, size_t count)
{
  auto execpol  = rmm::exec_policy(0);
  size_t kcount = pImpl->keys_count();
  int check =
    thrust::count_if(execpol->on(0), indexes, indexes + count, [kcount] __device__(int val) {
      return (val < 0) || (val >= kcount);
    });
  if (check > 0) throw std::out_of_range("gather: invalid index value");

  numeric_category<T>* result = new numeric_category<T>;
  result->pImpl->init_keys(pImpl->get_keys(), kcount);
  result->pImpl->bkeyset_includes_null = pImpl->bkeyset_includes_null;
  int* d_new_values                    = result->pImpl->get_values(count);
  thrust::copy(execpol->on(0), indexes, indexes + count, d_new_values);

  const BYTE* d_nulls = pImpl->get_nulls();
  if (d_nulls && indexes) {
    BYTE* d_new_nulls = result->pImpl->get_nulls(count);
    size_t byte_count = (count + 7) / 8;
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<size_t>(0),
                       byte_count,
                       gather_nullbits_fn{indexes, count, d_new_nulls});
  }
  return result;
}

template <typename T>
numeric_category<T>* numeric_category<T>::gather_values(const int* indexes, size_t count)
{
  auto execpol  = rmm::exec_policy(0);
  size_t vcount = pImpl->values_count();
  int check =
    thrust::count_if(execpol->on(0), indexes, indexes + count, [vcount] __device__(int val) {
      return (val < 0) || (val >= vcount);
    });
  if (check > 0) throw std::out_of_range("gather_values: invalid index value");

  numeric_category<T>* result = new numeric_category<T>;
  result->pImpl->init_keys(pImpl->get_keys(), pImpl->keys_count());
  result->pImpl->bkeyset_includes_null = pImpl->bkeyset_includes_null;
  const int* d_values                  = pImpl->get_values();
  int* d_new_values                    = result->pImpl->get_values(count);
  thrust::gather(execpol->on(0), indexes, indexes + count, d_values, d_new_values);

  const BYTE* d_nulls = pImpl->get_nulls();
  if (d_nulls && indexes) {
    BYTE* d_new_nulls = result->pImpl->get_nulls(count);
    size_t byte_count = (count + 7) / 8;
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<size_t>(0),
                       byte_count,
                       [d_nulls, indexes, count, d_new_nulls] __device__(size_t byte_idx) {
                         BYTE mask = 0;
                         for (int bit = 0; bit < 8; ++bit) {
                           size_t idx = byte_idx * 8 + bit;
                           if (idx < count) {
                             int flag = (int)(!is_item_null(d_nulls, indexes[idx]));
                             mask |= flag << bit;
                           }
                         }
                         d_new_nulls[byte_idx] = mask;
                       });
  }
  return result;
}
