---
name: build-test-cudf-java
description: Build and test cudf Java bindings (cudf-java) inside a cudf devcontainer. Use when the user asks to build, compile, or test Java code in the cudf repository.
---

# Build and Test cudf-java

## Prerequisites

### JDK and Maven

Ensure JDK (17+) and Maven are installed:

```bash
java -version && mvn --version
```

If either is missing, detect the OS and install using the appropriate package manager. Hints:

- **Debian/Ubuntu** (`apt`): `sudo apt-get update -qq && sudo apt-get install -y -qq default-jdk maven`
- **Fedora/RHEL** (`dnf`): `sudo dnf install -y java-17-openjdk-devel maven`
- **Arch** (`pacman`): `sudo pacman -S --noconfirm jdk17-openjdk maven`
- **macOS** (`brew`): `brew install openjdk@17 maven`

Detect the OS by checking which package manager is available (e.g. `command -v apt-get`, `command -v dnf`, etc.) and use the matching command.

### libcudf C++ with Java-required CMake flags

libcudf must be built with specific CMake flags for the Java bindings. Read `java/README.md` **(section: "Build From Source")** for the current required flags, then verify they are set:

```bash
grep -E "<FLAG1>|<FLAG2>|.." cpp/build/latest/CMakeCache.txt
```

If any flag is missing, reconfigure and rebuild libcudf following the `build-test-cudf` skill, passing the flags from the README. Ignore any flags that CMake reports as unused.

## Environment setup

Before any `mvn` command, export these variables from the `cudf/java` directory. All subsequent build and test commands assume these are set.

```bash
export CUDF_CPP_BUILD_DIR=$(readlink -f ../cpp/build/latest)
```

Export `MAVEN_OPTS` based on the JDK version. `--add-opens` flags are **required for every `mvn` invocation** on JDK 17+ (strong encapsulation). Without them, `gmaven-plugin:1.5` fails. On JDK 9-16 the flags are accepted but optional. On JDK 8 or below, **do not set them** — the JVM does not recognize `--add-opens` and will fail with `Unrecognized option`.

```bash
export MAVEN_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.util.regex=ALL-UNNAMED"
```

Export `MVN_COMMON_OPTS` to match the CI build configuration in `java/ci/build-in-docker.sh`. For example:

```bash
export MVN_COMMON_OPTS="-DCUDF_CPP_BUILD_DIR=$CUDF_CPP_BUILD_DIR -DBUILD_SHARED_LIBS=OFF -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON -DCUDA_STATIC_CUFILE=ON -DCUDF_JNI_LIBCUDF_STATIC=ON"
```

## Building cudf-java

The Java JNI native code must be compiled for the same CUDA architectures as libcudf. Detect what libcudf was built with:

```bash
grep CMAKE_CUDA_ARCHITECTURES cpp/build/latest/CMakeCache.txt
```

Update `MVN_COMMON_OPTS` to use that value for `-DCMAKE_CUDA_ARCHITECTURES` (by default use `NATIVE`).

```bash
export MVN_COMMON_OPTS=$MVN_COMMON_OPTS -DCMAKE_CUDA_ARCHITECTURES=<VALUE>
```

```bash
rm -rf target/cmake-build  # only needed if changing CMAKE_CUDA_ARCHITECTURES from a previous build
mvn install $MVN_COMMON_OPTS -DskipTests
```

Notes:
- Omit `rm -rf target/cmake-build` on incremental rebuilds when architectures haven't changed.
- The native compilation is the slow step. Subsequent runs reuse cached artifacts if `target/cmake-build` is preserved.

## Running Java tests

**Always run `mvn install -DskipTests` first** (see Building section above) before running tests. The `mvn test` goal re-triggers the cmake/native build step. If `target/cmake-build` already contains fully built artifacts from a prior `mvn install`, this is an incremental no-op. But if the cmake-build directory was cleaned or is missing, `mvn test` may hit a race condition where the linker tries to link `libcudfjni.so` before `libarrow.a` is fully built, causing a `cannot find libarrow.a` error. If this happens, re-run `mvn install -DskipTests` to rebuild the native code cleanly, then retry `mvn test`.

### All tests

```bash
mvn test $MVN_COMMON_OPTS
```

### Discovering tests

Java test sources are at `java/src/test/java/ai/rapids/cudf/`. Use `find` or `glob` to discover test classes:

```bash
find java/src/test/java -name "*Test.java" | head -20
```

### Specific test class

```bash
mvn test $MVN_COMMON_OPTS \
  -Dtest="ai.rapids.cudf.ast.<ClassName>" \
  -pl .
```

### Specific test method

```bash
mvn test $MVN_COMMON_OPTS \
  -Dtest="ai.rapids.cudf.ast.<ClassName>#<testName>" \
  -pl .
```

### Known pre-existing failures

`ArrowColumnVectorTest` may show errors on JDK 21+ due to Netty/Arrow module access restrictions. These are unrelated to cudf code changes.
