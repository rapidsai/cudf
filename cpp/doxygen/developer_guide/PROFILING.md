# Profiling libcudf

Profiling is essential for understanding performance characteristics and identifying bottlenecks in libcudf. This guide covers GPU profiling using NVIDIA Nsight Systems.

## NVIDIA Nsight Systems

[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) is a system-wide performance analysis tool that provides detailed timeline views of CPU and GPU activity.
It's the recommended tool for profiling CUDA applications and understanding kernel execution, memory transfers, and API calls.

### Installation

Nsight Systems is included with the CUDA Toolkit, or can be downloaded from https://developer.nvidia.com/nsight-systems. The command-line tool is `nsys`. Verify installation:

```bash
nsys --version
```

### Recommended Profile Command

When profiling cuDF workloads, use the following flags:

```bash
nsys profile --trace=nvtx,cuda,osrt --cuda-memory-usage=true --gpu-metrics-devices=0 --nvtx-domain-exclude=CCCL python script.py
```

**Options explained:**
- `--trace=nvtx,cuda,osrt`: Trace NVTX ranges, CUDA API calls, and OS runtime libraries
- `--cuda-memory-usage=true`: Track CUDA memory allocation and usage
- `--gpu-metrics-devices=0`: Collect GPU metrics from device 0
- `--nvtx-domain-exclude=CCCL`: Exclude verbose CCCL (CUDA C++ Core Libraries) NVTX ranges

### Profiling Specific GPUs

When working with multi-GPU systems, you may want to profile a specific GPU.
To profile GPUs other than device 0, use both `--gpu-metrics-devices=N` and `--env-var CUDA_VISIBLE_DEVICES=N` to ensure the application and profiler target the same device.

For example, modify the flags like this for profiling GPU 4:

```bash
nsys profile --trace=nvtx,cuda,osrt --cuda-memory-usage=true --gpu-metrics-devices=4 --env-var CUDA_VISIBLE_DEVICES=4 python script.py
```

### Analyzing Results

After profiling, open the `.nsys-rep` file in the Nsight Systems GUI to analyze CPU and GPU activity over time.
The interface shows individual kernel launches and durations, memory allocations and transfers, and metrics like memory bandwidth utilization.
