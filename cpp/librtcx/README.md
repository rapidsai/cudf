# librtcx

RTCX (runtime-compiler extended) is a wrapper around NVRTC and nvJitLink designed to provide:

- User-controlled compilation, linking, caching, and pre-loading of CUDA kernels
- Zero-copy interfaces to manage JIT compilation and linking
- CMake script to embed **compressed** headers directly into an executable without incurring overhead at runtime on every compilation request
- Facilities to pre-load and teardown dynamic library dependencies (`libcuda`, `libnvrtc`, and `libnvJitLink`)

## Platforms Supported
- Linux

## Build-time Requirements
- CMake >= 4.0
- LibZSTD
- xxHash
- CUDA >= 12.2

# Dependencies
- nvJitlink >= 12.2
- NVRTC >= 12.2
- LibCUDA >= 12.2
