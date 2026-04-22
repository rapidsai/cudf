# Doc

RTCX (runtime-compiler extended) is a wrapper around NVRTC and NVJitLink designed to provide:

- User-controlled compilation, linking, caching, and pre-loading of CUDA kernels
- Zero-copy interfaces to manage JIT compilation and linking
- CMake script to embed **compressed** headers directly into an executable without incurring overhead at runtime on every compilation request
- Facilities to pre-load and teardown dynamic library dependencies (`libcuda`, `libnvrtc`, and `libnvJitlink`)

## Platforms Supported
- Linux

## Build-Scripting Requirements
- CMake
- LibZSTD - for binary compression

## Runtime Requirements
- CUDA >= 11.8
