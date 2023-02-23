/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <Python.h>
#include <cudf/utilities/error.hpp>
#include <ios>
#include <stdexcept>

namespace cudf_python {
namespace exceptions {

/**
 * @brief Exception handler to map C++ exceptions to Python ones in Cython
 *
 * This exception handler extends the base exception handler provided by
 * Cython (https://github.com/cython/cython/blob/master/Cython/Utility/CppSupport.cpp#L9).
 * In addition to the exceptions that Cython itself supports, this file adds support
 * for additional exceptions thrown by libcudf that need to be mapped to specific Python
 * exceptions.
 *
 * Since this function interoperates with Python's exception state, it does not throw
 * any C++ exceptions.
 */
void cudf_exception_handler()
{
  // Catch a handful of different errors here and turn them into the
  // equivalent Python errors.
  try {
    if (PyErr_Occurred())
      ;  // let the latest Python exn pass through and ignore the current one else
    throw;
  } catch (const std::bad_alloc& exn) {
    PyErr_SetString(PyExc_MemoryError, exn.what());
  } catch (const std::bad_cast& exn) {
    PyErr_SetString(PyExc_TypeError, exn.what());
  } catch (const std::domain_error& exn) {
    PyErr_SetString(PyExc_ValueError, exn.what());
  } catch (const cudf::data_type_error& exn) {
    // Have to catch data_type_error before invalid_argument because it is a subclass
    PyErr_SetString(PyExc_TypeError, exn.what());
  } catch (const std::invalid_argument& exn) {
    PyErr_SetString(PyExc_ValueError, exn.what());
  } catch (const std::ios_base::failure& exn) {
    // Unfortunately, in standard C++ we have no way of distinguishing EOF
    // from other errors here; be careful with the exception mask
    PyErr_SetString(PyExc_IOError, exn.what());
  } catch (const std::out_of_range& exn) {
    // Change out_of_range to IndexError
    PyErr_SetString(PyExc_IndexError, exn.what());
  } catch (const std::overflow_error& exn) {
    PyErr_SetString(PyExc_OverflowError, exn.what());
  } catch (const std::range_error& exn) {
    PyErr_SetString(PyExc_ArithmeticError, exn.what());
  } catch (const std::underflow_error& exn) {
    PyErr_SetString(PyExc_ArithmeticError, exn.what());
    // The below is the default catch-all case.
  } catch (const std::exception& exn) {
    PyErr_SetString(PyExc_RuntimeError, exn.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
  }
}

}  // namespace exceptions
}  // namespace cudf_python
