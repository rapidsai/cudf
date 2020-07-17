/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;

/**
 * This class will load the native dependencies.
 */
public class NativeDepsLoader {
  private static final Logger log = LoggerFactory.getLogger(NativeDepsLoader.class);
  private static final String[] loadOrder = new String[] {
      "cudf",
      "cudfjni"
  };
  private static ClassLoader loader = NativeDepsLoader.class.getClassLoader();
  private static boolean loaded = false;

  /**
   * Load the native libraries needed for libcudf, if not loaded already.
   */
  public static synchronized void loadNativeDeps() {
    if (!loaded) {
      try {
        loadNativeDeps(loadOrder);
        loaded = true;
      } catch (Throwable t) {
        log.error("Could not load cudf jni library...", t);
      }
    }
  }

  /**
   * Allows other libraries to reuse the same native deps loading logic. Libraries will be searched
   * for under ${os.arch}/${os.name}/ in the class path using the class loader for this class. It
   * will also look for the libraries under ./target/native-deps/${os.arch}/${os.name} to help
   * facilitate testing while building.
   * <br/>
   * Because this just loads the libraries and loading the libraries themselves needs to be a
   * singleton operation it is recommended that any library using this provide their own wrapper
   * function similar to
   * <pre>
   *     private static boolean loaded = false;
   *     static synchronized void loadNativeDeps() {
   *         if (!loaded) {
   *             try {
   *                 // If you also depend on the cudf liobrary being loaded, be sure it is loaded
   *                 // first
   *                 ai.rapids.cudf.NativeDepsLoader.loadNativeDeps();
   *                 ai.rapids.cudf.NativeDepsLoader.loadNativeDeps(new String[]{...});
   *                 loaded = true;
   *             } catch (Throwable t) {
   *                 log.error("Could not load ...", t);
   *             }
   *         }
   *     }
   * </pre>
   * This function should be called from the static initialization block of any class that uses
   * JNI. For example
   * <pre>
   *     public class UsesJNI {
   *         static {
   *             MyNativeDepsLoader.loadNativeDeps();
   *         }
   *     }
   * </pre>
   * @param loadOrder the base name of the libraries. For example libfoo.so would be passed in as
   *                  "foo".  The libraries are loaded in the order provided.
   * @throws IOException on any error trying to load the libraries.
   */
  public static void loadNativeDeps(String[] loadOrder) throws IOException {
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");

    for (String toLoad : loadOrder) {
      loadDep(os, arch, toLoad);
    }
  }

  private static void loadDep(String os, String arch, String baseName) throws IOException {
    String path = arch + "/" + os + "/" + System.mapLibraryName(baseName);
    File loc;
    URL resource = loader.getResource(path);
    if (resource == null) {
      // It looks like we are not running from the jar, or there are issues with the jar
      File f = new File("./target/native-deps/" + path);
      if (!f.exists()) {
        throw new FileNotFoundException("Could not locate native dependency " + path);
      }
      resource = f.toURL();
    }
    try (InputStream in = resource.openStream()) {
      loc = File.createTempFile(baseName, ".so");
      try (OutputStream out = new FileOutputStream(loc)) {
        byte[] buffer = new byte[1024 * 16];
        int read = 0;
        while ((read = in.read(buffer)) >= 0) {
          out.write(buffer, 0, read);
        }
      }
    }
    loc.deleteOnExit();
    System.load(loc.getAbsolutePath());
    loc.delete();
  }

  public static boolean libraryLoaded() {
    if (!loaded) {
      loadNativeDeps();
    }
    return loaded;
  }
}
