#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <iostream>
#include <string>

namespace cudf {
namespace debug {

template <typename T>
void print(std::vector<T> const& vec,
           std::ostream& os             = std::cout,
           std::string const& delimiter = ",")
{
  std::vector<double> f64s(vec.size());
  std::copy(vec.begin(), vec.end(), f64s.begin());
  os << "size: " << vec.size() << " [" << std::endl << "  ";
  std::copy(f64s.begin(), f64s.end(), std::ostream_iterator<double>(os, delimiter.data()));
  os << std::endl << "]" << std::endl;
}

template <typename T>
void print(rmm::device_vector<T> const& vec,
           std::ostream& os             = std::cout,
           std::string const& delimiter = ",",
           cudaStream_t stream          = 0)
{
  CUDA_TRY(cudaStreamSynchronize(stream));
  std::vector<T> hvec(vec.size());
  std::fill(hvec.begin(), hvec.end(), T{0});
  thrust::copy(vec.begin(), vec.end(), hvec.begin());
  print<T>(hvec, os, delimiter);
}

template <typename T>
void print(rmm::device_uvector<T> const& uvec,
           std::ostream& os             = std::cout,
           std::string const& delimiter = ",",
           cudaStream_t stream          = 0)
{
  rmm::device_vector<T> dvec(uvec.size());
  std::fill(dvec.begin(), dvec.end(), T{0});
  thrust::copy(rmm::exec_policy(stream)->on(stream), uvec.begin(), uvec.end(), dvec.begin());
  print<T>(dvec, os, delimiter, stream);
}

template <typename T>
void print(rmm::device_buffer const& buf,
           std::ostream& os             = std::cout,
           std::string const& delimiter = ",",
           cudaStream_t stream          = 0)
{
  auto ptr = thrust::device_pointer_cast<const T>(buf.data());
  rmm::device_vector<T> dvec(buf.size() / sizeof(T));
  thrust::fill(dvec.begin(), dvec.end(), T{0});
  thrust::copy(rmm::exec_policy(stream)->on(stream), ptr, ptr + dvec.size(), dvec.begin());
  print<T>(dvec, os, delimiter, stream);
}

template <typename T>
void print(cudf::column_view const& col,
           std::ostream& os             = std::cout,
           std::string const& delimiter = ",",
           cudaStream_t stream          = 0)
{
  rmm::device_vector<T> dvec(col.size());
  std::fill(dvec.begin(), dvec.end(), T{0});
  thrust::copy(rmm::exec_policy(stream)->on(stream), col.begin<T>(), col.end<T>(), dvec.begin());
  print<T>(dvec, os, delimiter, stream);
}

template <typename T>
void print(thrust::device_ptr<T> const& ptr,
           cudf::size_type size,
           std::ostream& os             = std::cout,
           std::string const& delimiter = ",",
           cudaStream_t stream          = 0)
{
  rmm::device_vector<T> dvec(size);
  std::fill(dvec.begin(), dvec.end(), T{0});
  thrust::copy(rmm::exec_policy(stream)->on(stream), ptr, ptr + size, dvec.begin());
  print<T>(dvec, os, delimiter, stream);
}
}  // namespace debug
}  // namespace cudf
