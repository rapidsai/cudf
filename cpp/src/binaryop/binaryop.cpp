/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <binaryop/jit/code/code.h>
#include <jit/launcher.h>
#include <jit/parser.h>
#include <jit/type.h>
#include <binaryop/jit/util.hpp>
#include <cudf/datetime.hpp>  // replace eventually

#include "compiled/binary_ops.hpp"

#include <string>
#include <timestamps.hpp.jit>
#include <types.hpp.jit>
#include <libcudacxx/details/__config.jit>
#include <libcudacxx/simt/cfloat.jit>
#include <libcudacxx/simt/chrono.jit>
#include <libcudacxx/simt/ctime.jit>
#include <libcudacxx/simt/limits.jit>
#include <libcudacxx/simt/ratio.jit>
#include <libcudacxx/simt/type_traits.jit>
#include <libcudacxx/simt/version.jit>
#include <libcudacxx/libcxx/include/__config.jit>
#include <libcudacxx/libcxx/include/__undef_macros.jit>
#include <libcudacxx/libcxx/include/cfloat.jit>
#include <libcudacxx/libcxx/include/chrono.jit>
#include <libcudacxx/libcxx/include/ctime.jit>
#include <libcudacxx/libcxx/include/limits.jit>
#include <libcudacxx/libcxx/include/ratio.jit>
#include <libcudacxx/libcxx/include/type_traits.jit>

namespace cudf {
namespace experimental {

namespace binops {
namespace detail {
/**
 * @brief Computes output valid mask for op between a column and a scalar
 */
rmm::device_buffer scalar_col_valid_mask_and(
    column_view const& col,
    scalar const& s,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
  if (col.size() == 0) {
    return rmm::device_buffer{};
  }

  if (not s.is_valid()) {
    return create_null_mask(col.size(), mask_state::ALL_NULL, stream, mr);
  } else if (s.is_valid() && col.nullable()) {
    return copy_bitmask(col, stream, mr);
  } else {
    return rmm::device_buffer{};
  }
}
}  // namespace detail

namespace jit {

const std::string hash = "prog_binop.experimental";

const std::vector<std::string> compiler_flags{
    "-std=c++14",
    // suppress all NVRTC warnings
    "-w",
    // force libcudacxx to not include system headers
    "-D__CUDACC_RTC__",
    // __CHAR_BIT__ is from GCC, but libcxx uses it
    "-D__CHAR_BIT__=" + std::to_string(__CHAR_BIT__),
    // enable temporary workarounds to compile libcudacxx with nvrtc
    "-D_LIBCUDACXX_HAS_NO_CTIME",
    "-D_LIBCUDACXX_HAS_NO_WCHAR",
    "-D_LIBCUDACXX_HAS_NO_CFLOAT",
    "-D_LIBCUDACXX_HAS_NO_STDINT",
    "-D_LIBCUDACXX_HAS_NO_CSTDDEF",
    "-D_LIBCUDACXX_HAS_NO_CLIMITS",
    "-D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS",
};
const std::vector<std::string> header_names{
    "operation.h", "traits.h", cudf_types_hpp, cudf_wrappers_timestamps_hpp};

const std::unordered_map<std::string, char const *> stringified_headers{
  {"details/../../libcxx/include/__config", libcxx_config},
  {"../libcxx/include/__undef_macros", libcxx_undef_macros},
  {"simt/../../libcxx/include/cfloat", libcxx_cfloat},
  {"simt/../../libcxx/include/chrono", libcxx_chrono},
  {"simt/../../libcxx/include/ctime", libcxx_ctime},
  {"simt/../../libcxx/include/limits", libcxx_limits},
  {"simt/../../libcxx/include/ratio", libcxx_ratio},
  {"simt/../../libcxx/include/type_traits", libcxx_type_traits},
  {"simt/../details/__config", libcudacxx_details_config},
  {"simt/cfloat", libcudacxx_simt_cfloat},
  {"simt/chrono", libcudacxx_simt_chrono},
  {"simt/ctime", libcudacxx_simt_ctime},
  {"simt/limits", libcudacxx_simt_limits},
  {"simt/ratio", libcudacxx_simt_ratio},
  {"simt/type_traits", libcudacxx_simt_type_traits},
  {"simt/version", libcudacxx_simt_version},
};

std::istream* send_stringified_header(std::iostream& stream,
                                      char const* header) {
  // skip the filename line added by stringify
  stream << (std::strchr(header, '\n') + 1);
  return &stream;
}

std::istream* headers_code(std::string filename, std::iostream& stream) {
  if (filename == "operation.h") {
    stream << code::operation;
    return &stream;
  }
  if (filename == "traits.h") {
    stream << code::traits;
    return &stream;
  }
  auto it = stringified_headers.find(filename);
  if (it != stringified_headers.end()) {
    return send_stringified_header(stream, it->second);
  }
  return nullptr;
}

void binary_operation(mutable_column_view& out,
                      scalar const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      cudaStream_t stream) {
  cudf::jit::launcher(hash, code::kernel, header_names, compiler_flags,
                      headers_code, stream)
      .set_kernel_inst(
          "kernel_v_s",                           // name of the kernel we are
                                                  // launching
          {cudf::jit::get_type_name(out.type()),  // list of template arguments
           cudf::jit::get_type_name(rhs.type()),
           cudf::jit::get_type_name(lhs.type()),
           get_operator_name(op, OperatorType::Reverse)})
      .launch(out.size(), cudf::jit::get_data_ptr(out),
              cudf::jit::get_data_ptr(rhs), cudf::jit::get_data_ptr(lhs));
}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator op,
                      cudaStream_t stream) {
  cudf::jit::launcher(hash, code::kernel, header_names, compiler_flags,
                      headers_code, stream)
      .set_kernel_inst(
          "kernel_v_s",                           // name of the kernel we are
                                                  // launching
          {cudf::jit::get_type_name(out.type()),  // list of template arguments
           cudf::jit::get_type_name(lhs.type()),
           cudf::jit::get_type_name(rhs.type()),
           get_operator_name(op, OperatorType::Direct)})
      .launch(out.size(), cudf::jit::get_data_ptr(out),
              cudf::jit::get_data_ptr(lhs), cudf::jit::get_data_ptr(rhs));
}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      cudaStream_t stream) {
  cudf::jit::launcher(hash, code::kernel, header_names, compiler_flags,
                      headers_code, stream)
      .set_kernel_inst(
          "kernel_v_v",                           // name of the kernel we are
                                                  // launching
          {cudf::jit::get_type_name(out.type()),  // list of template arguments
           cudf::jit::get_type_name(lhs.type()),
           cudf::jit::get_type_name(rhs.type()),
           get_operator_name(op, OperatorType::Direct)})
      .launch(out.size(), cudf::jit::get_data_ptr(out),
              cudf::jit::get_data_ptr(lhs), cudf::jit::get_data_ptr(rhs));
}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      const std::string& ptx,
                      cudaStream_t stream) {
  std::string const output_type_name = cudf::jit::get_type_name(out.type());

  std::string ptx_hash =
      hash + "." +
      std::to_string(std::hash<std::string>{}(ptx + output_type_name));
  std::string cuda_source = "\n#include <cudf/types.hpp>\n" +
                            cudf::jit::parse_single_function_ptx(
                                ptx, "GENERIC_BINARY_OP", output_type_name) +
                            code::kernel;

  cudf::jit::launcher(ptx_hash, cuda_source, header_names, compiler_flags,
                      headers_code, stream)
      .set_kernel_inst("kernel_v_v",       // name of the kernel
                                           // we are launching
                       {output_type_name,  // list of template arguments
                        cudf::jit::get_type_name(lhs.type()),
                        cudf::jit::get_type_name(rhs.type()),
                        get_operator_name(binary_operator::GENERIC_BINARY,
                                          OperatorType::Direct)})
      .launch(out.size(), cudf::jit::get_data_ptr(out),
              cudf::jit::get_data_ptr(lhs), cudf::jit::get_data_ptr(rhs));
}

}  // namespace jit
}  // namespace binops

namespace detail {

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream) {
  // Check for datatype
  CUDF_EXPECTS(is_fixed_width(output_type),
               "Invalid/Unsupported output datatype");

  if ((lhs.type().id() == type_id::STRING) &&
      (rhs.type().id() == type_id::STRING)) {
    return binops::compiled::binary_operation(lhs, rhs, op, output_type, mr,
                                              stream);
  }

  CUDF_EXPECTS(is_fixed_width(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_fixed_width(rhs.type()), "Invalid/Unsupported rhs datatype");

  auto new_mask = binops::detail::scalar_col_valid_mask_and(rhs, lhs, stream, mr);
  auto out = make_numeric_column(output_type, rhs.size(), new_mask,
                                 cudf::UNKNOWN_NULL_COUNT, stream, mr);

  if (rhs.size() == 0) {
    return out;
  }

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  return out;
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream) {
  // Check for datatype
  CUDF_EXPECTS(is_fixed_width(output_type),
               "Invalid/Unsupported output datatype");

  if ((lhs.type().id() == type_id::STRING) &&
      (rhs.type().id() == type_id::STRING)) {
    return binops::compiled::binary_operation(lhs, rhs, op, output_type, mr,
                                              stream);
  }

  CUDF_EXPECTS(is_fixed_width(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_fixed_width(rhs.type()), "Invalid/Unsupported rhs datatype");

  auto new_mask = binops::detail::scalar_col_valid_mask_and(lhs, rhs, stream, mr);
  auto out = make_numeric_column(output_type, lhs.size(), new_mask,
                                 cudf::UNKNOWN_NULL_COUNT, stream, mr);

  if (lhs.size() == 0) {
    return out;
  }

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  return out;
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream) {
  CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  // Check for datatype
  CUDF_EXPECTS(is_fixed_width(output_type),
               "Invalid/Unsupported output datatype");

  if ((lhs.type().id() == type_id::STRING) &&
      (rhs.type().id() == type_id::STRING)) {
    return binops::compiled::binary_operation(lhs, rhs, op, output_type, mr,
                                              stream);
  }

  CUDF_EXPECTS(is_fixed_width(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_fixed_width(rhs.type()), "Invalid/Unsupported rhs datatype");

  auto new_mask = bitmask_and(lhs, rhs, stream, mr);
  auto out = make_fixed_width_column(output_type, lhs.size(), new_mask,
                                     cudf::UNKNOWN_NULL_COUNT, stream, mr);

  // Check for 0 sized data
  if (lhs.size() == 0 || rhs.size() == 0) {
    return out;
  }

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  return out;
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream) {
  // Check for datatype
  auto is_type_supported_ptx = [](data_type type) -> bool {
    return is_fixed_width(type) and
           type.id() != type_id::INT8;  // Numba PTX doesn't support int8
  };

  CUDF_EXPECTS(is_type_supported_ptx(lhs.type()),
               "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_type_supported_ptx(rhs.type()),
               "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_type_supported_ptx(output_type),
               "Invalid/Unsupported output datatype");

  CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  auto new_mask = bitmask_and(lhs, rhs, stream, mr);
  auto out = make_fixed_width_column(output_type, lhs.size(), new_mask,
                                     cudf::UNKNOWN_NULL_COUNT, stream, mr);

  // Check for 0 sized data
  if (lhs.size() == 0 || rhs.size() == 0) {
    return out;
  }

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ptx, stream);
  return out;
}

}  // namespace detail

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr) {
  return detail::binary_operation(lhs, rhs, op, output_type, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr) {
  return detail::binary_operation(lhs, rhs, op, output_type, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr) {
  return detail::binary_operation(lhs, rhs, op, output_type, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr) {
  return detail::binary_operation(lhs, rhs, ptx, output_type, mr);
}

}  // namespace experimental
}  // namespace cudf
