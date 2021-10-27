/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

/** Utility class that wraps an array of closeable instances and can be closed */
public class CloseableArray<T extends AutoCloseable> implements AutoCloseable {
  private T[] array;

  public static <T extends AutoCloseable> CloseableArray<T> wrap(T[] array) {
    return new CloseableArray<T>(array);
  }

  CloseableArray(T[] array) {
    this.array = array;
  }

  public int size() {
    return array.length;
  }

  public T get(int i) {
    return array[i];
  }

  public T set(int i, T obj) {
    array[i] = obj;
    return obj;
  }

  public T[] getArray() {
    return array;
  }

  public T[] release() {
    T[] result = array;
    array = null;
    return result;
  }

  public void closeAt(int i) {
    try {
      T toClose = array[i];
      array[i] = null;
      toClose.close();
    } catch (RuntimeException e) {
      throw e;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void close() {
    close(null);
  }

  public void close(Exception pendingError) {
    if (array == null) {
      return;
    }
    T[] toClose = array;
    array = null;
    RuntimeException error = null;
    if (pendingError instanceof RuntimeException) {
      error = (RuntimeException) pendingError;
    } else if (pendingError != null) {
      error = new RuntimeException(pendingError);
    }
    for (T obj: toClose) {
      if (obj != null) {
        try {
          obj.close();
        } catch (RuntimeException e) {
          if (error != null) {
            error.addSuppressed(e);
          } else {
            error = e;
          }
        } catch (Exception e) {
          if (error != null) {
            error.addSuppressed(e);
          } else {
            error = new RuntimeException(e);
          }
        }
      }
    }
    if (error != null) {
      throw error;
    }
  }
}
