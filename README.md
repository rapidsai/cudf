# GPU open analytics initiative

Continuum Analytics, H2O.ai, and MapD Technologies have announced the formation of the GPU Open Analytics Initiative (GOAI) to create common data frameworks enabling developers and statistical researchers to accelerate data science on GPUs. GOAI will foster the development of a data science ecosystem on GPUs by allowing resident applications to interchange data seamlessly and efficiently. 

![GOAI](img/goai_logo_3.png)

## GPU Data Frame

Our first project: an open source GPU Data Frame with a corresponding Python API. The GPU Data Frame is a common API that enables efficient interchange of data between processes running on the GPU. End-to-end computation on the GPU avoids transfers back to the CPU or copying of in-memory data reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. Users of the MapD Core database can output the results of a SQL query into the GPU Data Frame, which then can be manipulated by the Continuum Analytics’ Anaconda NumPy-like Python API or used as input into the H2O suite of machine learning algorithms without additional data manipulation.

![Architecture](img/GPU_df_arch_diagram.png)

Users of the MapD Core database can output the results of a SQL query into the GPU Data Frame, which then can be manipulated by the Continuum Analytics’ Anaconda NumPy-like Python API or used as input into the H2O suite of machine learning algorithms without additional data manipulation. In early internal tests, this approach exhibited order-of-magnitude improvements in processing times compared to passing the data between applications on a CPU. 

![Architecture](img/mapd-conda-h2o.png)
