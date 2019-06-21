/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
 * Exception from the cuda language/library.  Be aware that because of how cuda does asynchronous
 * processing exceptions from cuda can be thrown by method calls that did not cause the exception
 * to take place.  These will take place on the same thread that caused the error.
 * <p>
 * Please See
 * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html">the cuda docs</a>
 * for more details on how this works.
 * <p>
 * In general you can recover from cuda errors even in async calls if you make sure that you
 * don't switch between threads for different parts of processing that can be retried as a chunk.
 */
public class CudaException extends RuntimeException {
  CudaException(String message) {
    super(message);
  }

  CudaException(String message, Throwable cause) {
    super(message, cause);
  }
}
