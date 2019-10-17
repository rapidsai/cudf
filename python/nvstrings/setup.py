import os

from cmake_setuptools import CMakeBuildExt, CMakeExtension
from setuptools import setup

install_requires = []

cuda_version = "".join(
    os.environ.get("CUDA_VERSION", "unknown").split(".")[:2]
)
name = "nvstrings-cuda{}".format(cuda_version)
version = os.environ.get("GIT_DESCRIBE_TAG", "0.0.0.dev0").lstrip("v")

setup(
    name=name,
    description="CUDA strings Python bindings",
    url="https://github.com/rapidsai/cudf",
    version=version,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    py_modules=["nvstrings", "nvcategory", "nvtext"],
    author="NVIDIA Corporation",
    license="Apache",
    install_requires=install_requires,
    ext_modules=[
        CMakeExtension("pyniNVStrings", "cpp"),
        CMakeExtension("pyniNVCategory", "cpp"),
        CMakeExtension("pyniNVText", "cpp"),
    ],
    cmdclass={"build_ext": CMakeBuildExt},
    zip_safe=False,
)
