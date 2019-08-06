1. cuStreamz is a GPU-accelerated Streaming Library, which uses cuDF with Streamz for stream data processing on GPUs. 
2. cuStreamz has its own conda metapackage which makes it as simple as possible to install the set of dependencies necessary to process streaming workloads on GPUs.
3. A series of tests for use in a cuDF gpuCI instance have been included ensuring that changes continuously rolled out as part of cuDF don't break its integration with Streamz.
4. You can find example notebooks on how to write cuStreamz jobs in the RAPIDS notebooks-extended repository.  
