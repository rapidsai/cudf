import os
import shutil
import sysconfig
from distutils.sysconfig import get_python_lib

import numpy
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

# Locate CUDA_HOME
CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))

cuda_include_dir = os.path.join(CUDA_HOME, "include")
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")


# Include the ability to compile using both GCC and NVCC
def compile_cuda(self):
    self.src_extensions.append(".cu")
    default_so = self.compiler_so
    super = self._compile

    # Adjust the _compile method.
    def _compile(obj, src, ext, cc_args, extra_args, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # cude source file, lets compile it accordingly
            self.set_executable(
                "compiler_so",
                os.path.join(os.path.join(CUDA_HOME, "bin"), "nvcc"),
            )
            postargs = extra_args["nvcc"]
        else:
            postargs = extra_args["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)

        # Reset defaults
        self.compiler_so = default_so

    self._compile = _compile


# Stub for custom compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        compile_cuda(self.compiler)
        build_ext.build_extensions(self)


try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


ext = Extension(
    "cudfkernel",
    sources=["src/kernel_wrapper.cu", "kernel.pyx"],
    library_dirs=[
        cuda_lib_dir,
        get_python_lib(),
        os.path.join(os.sys.prefix, "lib"),
    ],
    libraries=["cudf", "cudart"],
    language="c++",
    runtime_library_dirs=[cuda_lib_dir],
    include_dirs=[
        os.path.dirname(sysconfig.get_path("include")),
        numpy_include,
        cuda_include_dir,
        "src",
        "../../../cpp/thirdparty/libcudacxx/include",
        "../../../cpp/thirdparty/cub",
    ],
    extra_compile_args={
        "gcc": [],
        "nvcc": ["--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'"],
    },
)

setup(
    name="cudfkernel",
    version="0.1",
    url="https://github.com/rapidsai/cudf",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Programming Language :: Python",
    ],
    ext_modules=[ext],
    packages=find_packages(include=["cudf", "cudf.*"]),
    cmdclass={"build_ext": custom_build_ext},
    zip_safe=False,
)
