/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This class will load the native dependencies.
 */
public class NativeDepsLoader {
  private static final Logger log = LoggerFactory.getLogger(NativeDepsLoader.class);

  /**
   * Set this system property to true to prevent unpacked dependency files from
   * being deleted immediately after they are loaded. The files will still be
   * scheduled for deletion upon exit.
   */
  private static final Boolean preserveDepsAfterLoad = Boolean.getBoolean(
      "ai.rapids.cudf.preserve-dependencies");

  /**
   * Defines the loading order for the dependencies. Dependencies are loaded in
   * stages where all the dependencies in a stage are not interdependent and
   * therefore can be loaded in parallel. All dependencies within an earlier
   * stage are guaranteed to have finished loading before any dependencies in
   * subsequent stages are loaded.
   */
  private static final String[][] loadOrder = new String[][]{
      new String[]{
          "nvcomp"
      },
      new String[]{
          "cudf"
      },
      new String[]{
          "cudfjni"
      }
  };
  private static final ClassLoader loader = NativeDepsLoader.class.getClassLoader();

  private static boolean loaded = false;

  /**
   * Load the native libraries needed for libcudf, if not loaded already.
   */
  public static synchronized void loadNativeDeps() {
    if (!loaded) {
      try {
        loadNativeDeps(loadOrder, preserveDepsAfterLoad);
        loaded = true;
      } catch (Throwable t) {
        log.error("Could not load cudf jni library...", t);
      }
    }
  }

  /**
   * Allows other libraries to reuse the same native deps loading logic. Libraries will be searched
   * for under ${os.arch}/${os.name}/ in the class path using the class loader for this class.
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
    loadNativeDeps(loadOrder, preserveDepsAfterLoad);
  }

  /**
   * Allows other libraries to reuse the same native deps loading logic. Libraries will be searched
   * for under ${os.arch}/${os.name}/ in the class path using the class loader for this class.
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
   * @param preserveDeps if false the dependencies will be deleted immediately after loading
   *                     rather than on exit.
   * @throws IOException on any error trying to load the libraries.
   */
  public static void loadNativeDeps(String[] loadOrder, boolean preserveDeps) throws IOException {
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");

    for (String toLoad : loadOrder) {
      loadDep(os, arch, toLoad, preserveDeps);
    }
  }

  /**
   * Optionally load native dependencies. This method attempts to load the specified libraries
   * but does not throw exceptions on failure. Instead, it returns true if all libraries were
   * loaded successfully, false otherwise.
   * @param loadOrder the base name of the libraries. For example libfoo.so would be passed in as
   *                  "foo". The libraries are loaded in the order provided.
   * @return true if all libraries were loaded successfully, false otherwise
   */
  public static boolean loadOptionalNativeDeps(String[] loadOrder) {
    try {
      loadNativeDeps(loadOrder, preserveDepsAfterLoad);
      return true;
    } catch (Throwable t) {
      log.warn("Could not load optional native dependencies: " + t.getMessage());
      return false;
    }
  }

  /**
   * Load native dependencies in stages, where the dependency libraries in each stage
   * are loaded only after all libraries in earlier stages have completed loading.
   * @param loadOrder array of stages with an array of dependency library names in each stage
   * @param preserveDeps if false the dependencies will be deleted immediately after loading
   *                     rather than on exit.
   * @throws IOException on any error trying to load the libraries
   */
  private static void loadNativeDeps(String[][] loadOrder, boolean preserveDeps) throws IOException {
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");

    ExecutorService executor = Executors.newCachedThreadPool();
    List<List<Future<File>>> allFileFutures = new ArrayList<>();

    // Start unpacking and creating the temporary files for each dependency.
    // Unpacking a dependency does not depend on stage order.
    for (String[] stageDependencies : loadOrder) {
      List<Future<File>> stageFileFutures = new ArrayList<>();
      allFileFutures.add(stageFileFutures);
      for (String name : stageDependencies) {
        stageFileFutures.add(executor.submit(() -> createFile(os, arch, name)));
      }
    }

    List<Future<?>> loadCompletionFutures = new ArrayList<>();

    // Proceed stage-by-stage waiting for the dependency file to have been
    // produced then submit them to the thread pool to be loaded.
    for (List<Future<File>> stageFileFutures : allFileFutures) {
      // Submit all dependencies in the stage to be loaded in parallel
      loadCompletionFutures.clear();
      for (Future<File> fileFuture : stageFileFutures) {
        loadCompletionFutures.add(executor.submit(() -> loadDep(fileFuture, preserveDeps)));
      }

      // Wait for all dependencies in this stage to have been loaded
      for (Future<?> loadCompletionFuture : loadCompletionFutures) {
        try {
          loadCompletionFuture.get();
        } catch (ExecutionException | InterruptedException e) {
          throw new IOException("Error loading dependencies", e);
        }
      }
    }

    executor.shutdownNow();
  }

  /**
   * Allows other libraries to reuse the same native deps loading logic. Library will be searched
   * for under ${os.arch}/${os.name}/ in the class path using the class loader for this class.
   * @param depName the base name of the library. For example libfoo.so would be passed in as
   *                "foo".  The libraries are loaded in the order provided.
   * @param preserveDep if false the dependencies will be deleted immediately after loading
   *                    rather than on exit.
   * @return path where the dependency was loaded
   * @throws IOException on any error trying to load the libraries.
   */
  public static File loadNativeDep(String depName, boolean preserveDep) throws IOException {
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");
    return loadDep(os, arch, depName, preserveDep);
  }

  private static File loadDep(String os, String arch, String baseName, boolean preserveDep)
      throws IOException {
    File path = createFile(os, arch, baseName);
    loadDep(path, preserveDep);
    return path;
  }

  /** Load a library at the specified path */
  private static void loadDep(File path, boolean preserveDep) {
    System.load(path.getAbsolutePath());
    if (!preserveDep) {
      path.delete();
    }
  }

  /** Load a library, waiting for the specified future to produce the path before loading */
  private static void loadDep(Future<File> fileFuture, boolean preserveDep) {
    File path;
    try {
      path = fileFuture.get();
    } catch (ExecutionException | InterruptedException e) {
      throw new RuntimeException("Error loading dependencies", e);
    }
    loadDep(path, preserveDep);
  }

  /** Extract the contents of a library resource into a temporary file */
  private static File createFile(String os, String arch, String baseName) throws IOException {
    String path = arch + "/" + os + "/" + System.mapLibraryName(baseName);
    File loc;
    URL resource = loader.getResource(path);
    if (resource == null) {
      throw new FileNotFoundException("Could not locate native dependency " + path);
    }
    try (InputStream in = resource.openStream()) {
      loc = File.createTempFile(baseName, ".so");
      loc.deleteOnExit();
      try (OutputStream out = new FileOutputStream(loc)) {
        byte[] buffer = new byte[1024 * 16];
        int read = 0;
        while ((read = in.read(buffer)) >= 0) {
          out.write(buffer, 0, read);
        }
      }
    }
    return loc;
  }

  public static boolean libraryLoaded() {
    if (!loaded) {
      loadNativeDeps();
    }
    return loaded;
  }
}
