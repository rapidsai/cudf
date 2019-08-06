from distributed import Client, LocalCluster
#This creates a local Dask cluster with a scheduler on port 8786, and 24 single-threaded Dask workers.
cluster = LocalCluster(ip='0.0.0.0', n_workers=24, threads_per_worker=1, processes=True, asynchronous=True, scheduler_port=8786)
cluster
