/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
package ai.rapids.cudf;

/**
 * Exception thrown by cudf itself.
 */
public class CudfException extends RuntimeException {
  CudfException(String message) {
    super(message);
  }

  CudfException(String message, String nativeStacktrace) {
    super(getExceptionMessage(message, nativeStacktrace));
  }

  CudfException(String message, String nativeStacktrace, Throwable cause) {
    super(getExceptionMessage(message, nativeStacktrace), cause);
  }

  private static String getExceptionMessage(String message, String nativeStacktrace) {
    if (nativeStacktrace == null) {
      return message;
    }
    return message + "\n\t========== native stack frame ==========\n" + nativeStacktrace;
  }
}
