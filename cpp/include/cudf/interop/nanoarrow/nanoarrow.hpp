// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <stdexcept>
#include <string>
#include <vector>

#include "nanoarrow.h"

#ifndef NANOARROW_HPP_INCLUDED
#define NANOARROW_HPP_INCLUDED

/// \defgroup nanoarrow_hpp Nanoarrow C++ Helpers
///
/// The utilities provided in this file are intended to support C++ users
/// of the nanoarrow C library such that C++-style resource allocation
/// and error handling can be used with nanoarrow data structures.
/// These utilities are not intended to mirror the nanoarrow C API.

namespace nanoarrow {

/// \defgroup nanoarrow_hpp-errors Error handling helpers
///
/// Most functions in the C API return an ArrowErrorCode to communicate
/// possible failure. Except where documented, it is usually not safe to
/// continue after a non-zero value has been returned. While the
/// nanoarrow C++ helpers do not throw any exceptions of their own,
/// these helpers are provided to facilitate using the nanoarrow C++ helpers
/// in frameworks where this is a useful error handling idiom.
///
/// @{

class Exception : public std::exception {
 public:
  Exception(const std::string& msg) : msg_(msg) {}
  const char* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};

#if defined(NANOARROW_DEBUG)
#define _NANOARROW_THROW_NOT_OK_IMPL(NAME, EXPR, EXPR_STR)                      \
  do {                                                                          \
    const int NAME = (EXPR);                                                    \
    if (NAME) {                                                                 \
      throw nanoarrow::Exception(                                               \
          std::string(EXPR_STR) + std::string(" failed with errno ") +          \
          std::to_string(NAME) + std::string("\n * ") + std::string(__FILE__) + \
          std::string(":") + std::to_string(__LINE__) + std::string("\n"));     \
    }                                                                           \
  } while (0)
#else
#define _NANOARROW_THROW_NOT_OK_IMPL(NAME, EXPR, EXPR_STR)            \
  do {                                                                \
    const int NAME = (EXPR);                                          \
    if (NAME) {                                                       \
      throw nanoarrow::Exception(std::string(EXPR_STR) +              \
                                 std::string(" failed with errno ") + \
                                 std::to_string(NAME));               \
    }                                                                 \
  } while (0)
#endif

#define NANOARROW_THROW_NOT_OK(EXPR)                                                   \
  _NANOARROW_THROW_NOT_OK_IMPL(_NANOARROW_MAKE_NAME(errno_status_, __COUNTER__), EXPR, \
                               #EXPR)

/// @}

namespace internal {

/// \defgroup nanoarrow_hpp-unique_base Base classes for Unique wrappers
///
/// @{

template <typename T>
static inline void init_pointer(T* data);

template <typename T>
static inline void move_pointer(T* src, T* dst);

template <typename T>
static inline void release_pointer(T* data);

template <>
inline void init_pointer(struct ArrowSchema* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowSchema* src, struct ArrowSchema* dst) {
  ArrowSchemaMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowSchema* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowArray* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowArray* src, struct ArrowArray* dst) {
  ArrowArrayMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowArray* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowArrayStream* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowArrayStream* src, struct ArrowArrayStream* dst) {
  ArrowArrayStreamMove(src, dst);
}

template <>
inline void release_pointer(ArrowArrayStream* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowBuffer* data) {
  ArrowBufferInit(data);
}

template <>
inline void move_pointer(struct ArrowBuffer* src, struct ArrowBuffer* dst) {
  ArrowBufferMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowBuffer* data) {
  ArrowBufferReset(data);
}

template <>
inline void init_pointer(struct ArrowBitmap* data) {
  ArrowBitmapInit(data);
}

template <>
inline void move_pointer(struct ArrowBitmap* src, struct ArrowBitmap* dst) {
  ArrowBitmapMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowBitmap* data) {
  ArrowBitmapReset(data);
}

template <>
inline void init_pointer(struct ArrowArrayView* data) {
  ArrowArrayViewInitFromType(data, NANOARROW_TYPE_UNINITIALIZED);
}

template <>
inline void move_pointer(struct ArrowArrayView* src, struct ArrowArrayView* dst) {
  ArrowArrayViewMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowArrayView* data) {
  ArrowArrayViewReset(data);
}

/// \brief A unique_ptr-like base class for stack-allocatable objects
/// \tparam T The object type
template <typename T>
class Unique {
 public:
  /// \brief Construct an invalid instance of T holding no resources
  Unique() { init_pointer(&data_); }

  /// \brief Move and take ownership of data
  Unique(T* data) { move_pointer(data, &data_); }

  /// \brief Move and take ownership of data wrapped by rhs
  Unique(Unique&& rhs) : Unique(rhs.get()) {}
  Unique& operator=(Unique&& rhs) {
    reset(rhs.get());
    return *this;
  }

  // These objects are not copyable
  Unique(const Unique& rhs) = delete;

  /// \brief Get a pointer to the data owned by this object
  T* get() noexcept { return &data_; }
  const T* get() const noexcept { return &data_; }

  /// \brief Use the pointer operator to access fields of this object
  T* operator->() noexcept { return &data_; }
  const T* operator->() const noexcept { return &data_; }

  /// \brief Call data's release callback if valid
  void reset() { release_pointer(&data_); }

  /// \brief Call data's release callback if valid and move ownership of the data
  /// pointed to by data
  void reset(T* data) {
    reset();
    move_pointer(data, &data_);
  }

  /// \brief Move ownership of this object to the data pointed to by out
  void move(T* out) { move_pointer(&data_, out); }

  ~Unique() { reset(); }

 protected:
  T data_;
};

/// @}

}  // namespace internal

/// \defgroup nanoarrow_hpp-unique Unique object wrappers
///
/// The Arrow C Data interface, the Arrow C Stream interface, and the
/// nanoarrow C library use stack-allocatable objects, some of which
/// require initialization or cleanup.
///
/// @{

/// \brief Class wrapping a unique struct ArrowSchema
using UniqueSchema = internal::Unique<struct ArrowSchema>;

/// \brief Class wrapping a unique struct ArrowArray
using UniqueArray = internal::Unique<struct ArrowArray>;

/// \brief Class wrapping a unique struct ArrowArrayStream
using UniqueArrayStream = internal::Unique<struct ArrowArrayStream>;

/// \brief Class wrapping a unique struct ArrowBuffer
using UniqueBuffer = internal::Unique<struct ArrowBuffer>;

/// \brief Class wrapping a unique struct ArrowBitmap
using UniqueBitmap = internal::Unique<struct ArrowBitmap>;

/// \brief Class wrapping a unique struct ArrowArrayView
using UniqueArrayView = internal::Unique<struct ArrowArrayView>;

/// @}

/// \defgroup nanoarrow_hpp-array-stream ArrayStream helpers
///
/// These classes provide simple ArrowArrayStream implementations that
/// can be extended to help simplify the process of creating a valid
/// ArrowArrayStream implementation or used as-is for testing.
///
/// @{

/// @brief Export an ArrowArrayStream from a standard C++ class
/// @tparam T A class with methods `int GetSchema(ArrowSchema*)`, `int
/// GetNext(ArrowArray*)`, and `const char* GetLastError()`
///
/// This class allows a standard C++ class to be exported to a generic ArrowArrayStream
/// consumer by mapping C callback invocations to method calls on an instance of the
/// object whose lifecycle is owned by the ArrowArrayStream. See VectorArrayStream for
/// minimal useful example of this pattern.
///
/// The methods must be accessible to the ArrayStreamFactory, either as public methods or
/// by declaring ArrayStreamFactory<ImplClass> a friend. Implementors are encouraged (but
/// not required) to implement a ToArrayStream(ArrowArrayStream*) that creates a new
/// instance owned by the ArrowArrayStream and moves the relevant data to that instance.
///
/// An example implementation might be:
///
/// \code
/// class StreamImpl {
///  public:
///   // Public methods (e.g., constructor) used from C++ to initialize relevant data
///
///   // Idiomatic exporter to move data + lifecycle responsibility to an instance
///   // managed by the ArrowArrayStream callbacks
///   void ToArrayStream(struct ArrowArrayStream* out) {
///     ArrayStreamFactory<StreamImpl>::InitArrayStream(new StreamImpl(...), out);
///   }
///
///  private:
///   // Make relevant methods available to the ArrayStreamFactory
///   friend class ArrayStreamFactory<StreamImpl>;
///
///   // Method implementations (called from C, not normally interacted with from C++)
///   int GetSchema(struct ArrowSchema* schema) { return ENOTSUP; }
///   int GetNext(struct ArrowArray* array) { return ENOTSUP; }
///   const char* GetLastError() { nullptr; }
/// };
/// \endcode
///
/// An example usage might be:
///
/// \code
/// // Call constructor and/or public methods to initialize relevant data
/// StreamImpl impl;
///
/// // Export to ArrowArrayStream after data are finalized
/// UniqueArrayStream stream;
/// impl.ToArrayStream(stream.get());
/// \endcode
template <typename T>
class ArrayStreamFactory {
 public:
  /// \brief Take ownership of instance and populate callbacks of out
  static void InitArrayStream(T* instance, struct ArrowArrayStream* out) {
    out->get_schema = &get_schema_wrapper;
    out->get_next = &get_next_wrapper;
    out->get_last_error = &get_last_error_wrapper;
    out->release = &release_wrapper;
    out->private_data = instance;
  }

 private:
  static int get_schema_wrapper(struct ArrowArrayStream* stream,
                                struct ArrowSchema* schema) {
    return reinterpret_cast<T*>(stream->private_data)->GetSchema(schema);
  }

  static int get_next_wrapper(struct ArrowArrayStream* stream, struct ArrowArray* array) {
    return reinterpret_cast<T*>(stream->private_data)->GetNext(array);
  }

  static const char* get_last_error_wrapper(struct ArrowArrayStream* stream) {
    return reinterpret_cast<T*>(stream->private_data)->GetLastError();
  }

  static void release_wrapper(struct ArrowArrayStream* stream) {
    delete reinterpret_cast<T*>(stream->private_data);
    stream->release = nullptr;
    stream->private_data = nullptr;
  }
};

/// \brief An empty array stream
///
/// This class can be constructed from an struct ArrowSchema and implements a default
/// get_next() method that always marks the output ArrowArray as released.
///
/// DEPRECATED (0.4.0): Early versions of nanoarrow allowed subclasses to override
/// get_schema(), get_next(), and get_last_error(). This functionality will be removed
/// in a future release: use the pattern documented in ArrayStreamFactory to create
/// custom ArrowArrayStream implementations.
class EmptyArrayStream {
 public:
  /// \brief Create an EmptyArrayStream from an ArrowSchema
  ///
  /// Takes ownership of schema.
  EmptyArrayStream(struct ArrowSchema* schema) : schema_(schema) {
    ArrowErrorInit(&error_);
  }

  /// \brief Export to ArrowArrayStream
  void ToArrayStream(struct ArrowArrayStream* out) {
    EmptyArrayStream* impl = new EmptyArrayStream(schema_.get());
    ArrayStreamFactory<EmptyArrayStream>::InitArrayStream(impl, out);
  }

  /// \brief Create an empty UniqueArrayStream from a struct ArrowSchema
  ///
  /// DEPRECATED (0.4.0): Use the constructor + ToArrayStream() to export an
  /// EmptyArrayStream to an ArrowArrayStream consumer.
  static UniqueArrayStream MakeUnique(struct ArrowSchema* schema) {
    UniqueArrayStream stream;
    EmptyArrayStream(schema).ToArrayStream(stream.get());
    return stream;
  }

  virtual ~EmptyArrayStream() {}

 protected:
  UniqueSchema schema_;
  struct ArrowError error_;

  void MakeStream(struct ArrowArrayStream* stream) { ToArrayStream(stream); }

  virtual int get_schema(struct ArrowSchema* schema) {
    return ArrowSchemaDeepCopy(schema_.get(), schema);
  }

  virtual int get_next(struct ArrowArray* array) {
    array->release = nullptr;
    return NANOARROW_OK;
  }

  virtual const char* get_last_error() { return error_.message; }

 private:
  friend class ArrayStreamFactory<EmptyArrayStream>;

  int GetSchema(struct ArrowSchema* schema) { return get_schema(schema); }

  int GetNext(struct ArrowArray* array) { return get_next(array); }

  const char* GetLastError() { return get_last_error(); }
};

/// \brief Implementation of an ArrowArrayStream backed by a vector of UniqueArray objects
class VectorArrayStream {
 public:
  /// \brief Create a VectorArrayStream from an ArrowSchema + vector of UniqueArray
  ///
  /// Takes ownership of schema and moves arrays if possible.
  VectorArrayStream(struct ArrowSchema* schema, std::vector<UniqueArray> arrays)
      : offset_(0), schema_(schema), arrays_(std::move(arrays)) {}

  /// \brief Create a one-shot VectorArrayStream from an ArrowSchema + ArrowArray
  ///
  /// Takes ownership of schema and array.
  VectorArrayStream(struct ArrowSchema* schema, struct ArrowArray* array)
      : offset_(0), schema_(schema) {
    arrays_.emplace_back(array);
  }

  /// \brief Export to ArrowArrayStream
  void ToArrayStream(struct ArrowArrayStream* out) {
    VectorArrayStream* impl = new VectorArrayStream(schema_.get(), std::move(arrays_));
    ArrayStreamFactory<VectorArrayStream>::InitArrayStream(impl, out);
  }

  /// \brief Create a UniqueArrowArrayStream from an existing array
  ///
  /// DEPRECATED (0.4.0): Use the constructors + ToArrayStream() to export a
  /// VectorArrayStream to an ArrowArrayStream consumer.
  static UniqueArrayStream MakeUnique(struct ArrowSchema* schema,
                                      struct ArrowArray* array) {
    UniqueArrayStream stream;
    VectorArrayStream(schema, array).ToArrayStream(stream.get());
    return stream;
  }

  /// \brief Create a UniqueArrowArrayStream from existing arrays
  ///
  /// DEPRECATED (0.4.0): Use the constructor + ToArrayStream() to export a
  /// VectorArrayStream to an ArrowArrayStream consumer.
  static UniqueArrayStream MakeUnique(struct ArrowSchema* schema,
                                      std::vector<UniqueArray> arrays) {
    UniqueArrayStream stream;
    VectorArrayStream(schema, std::move(arrays)).ToArrayStream(stream.get());
    return stream;
  }

 private:
  int64_t offset_;
  UniqueSchema schema_;
  std::vector<UniqueArray> arrays_;

  friend class ArrayStreamFactory<VectorArrayStream>;

  int GetSchema(struct ArrowSchema* schema) {
    return ArrowSchemaDeepCopy(schema_.get(), schema);
  }

  int GetNext(struct ArrowArray* array) {
    if (offset_ < static_cast<int64_t>(arrays_.size())) {
      arrays_[offset_++].move(array);
    } else {
      array->release = nullptr;
    }

    return NANOARROW_OK;
  }

  const char* GetLastError() { return ""; }
};

/// @}

}  // namespace nanoarrow

#endif
