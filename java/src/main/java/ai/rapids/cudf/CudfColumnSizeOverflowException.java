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
package ai.rapids.cudf;

/**
 * Exception thrown when CUDF operation results in a column size
 * exceeding CUDF column size limits
 */
public class CudfColumnSizeOverflowException extends CudfException {
  CudfColumnSizeOverflowException(String message) {
    super(message);
  }

  CudfColumnSizeOverflowException(String message, String nativeStacktrace) {
    super(message, nativeStacktrace);
  }

  CudfColumnSizeOverflowException(String message, String nativeStacktrace, Throwable cause) {
    super(message, nativeStacktrace, cause);
  }
}
