**cuStreamz Kubernetes Setup**

*"The following instruction set is for setting up the cuStreamz pipeline on Google Kubernetes Engine (GKE). But, the overall process for deploying cuStreamz 
on any Kubernetes cluster would be the same."*
 — cuStreamz team


**Step 1: Creating a Kubernetes Cluster**
1. Create a single-node Kubernetes cluster using the GPU-accelerated Computing option on the Google Kubernetes Engine console page. 
2. We are using 32 vCPU cores and 2 NVIDIA T4 GPUs for the node. 

**Step 2: Create and push the component Docker containers to Google Cloud's Container Registry**
1. Install gcloud by following the instructions here: [https://cloud.google.com/sdk/docs/downloads-apt-get](https://cloud.google.com/sdk/docs/downloads-apt-get)
2. Configure authorization permissions for gclud using: `gcloud auth configure-docker`. This might prompt you to follow-up with: `gcloud auth login`
3. Now we are all set to build Docker containers from the Dockerfiles included in this repository and push them to the container registry. 
4. Go to each of the four folders having the Dockerfiles and use: `docker build . -t gcr.io/<name of Google Cloud Project>/<name of the image>:<version>`
   For example, `sudo docker build . -t gcr.io/nv-ai-infra/cc-custreamz_dask_client:0.1`. 
5. Then, push this container to the container registry using: `docker push gcr.io/<name of GC project>/<name of the image>:<version>`
   For example, `docker push gcr.io/nv-ai-infra/cc-custreamz_dask_client:0.1`.


**Step 3: Connect to your GKE cluster using Cloud Shell**
1. Connect to your cluster using the cloud shell option on your cluster in GKE.
2. Install GPU drivers on your Kubernetes cluster following the instructions on:
[https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers)
3. You may need to get credentials to install drivers. You can use: `gcloud container clusters get-credentials <name of your cluster> —zone=<zone>` to do this.
4. Check if driver has been correctly installed using: `kubectl get pods -n kube-system`

**Step 4: Set up the cuStreamz pipeline**
1. You can now set up the entire pipeline. But first, copy the .yml files included in this repository to your cluster. 
   **Please ensure to pull in the correct docker images in the .yml files.**
2. Before creating the deployments, you would need to open a few ports (30040-30043) on your cluster to access the Dask diagnostics dashboard, Jupyter endpoints for both the Dask scheduler 
   and the cuStreamz-Dask client. On Google Cloud, you can create a firewall rule on the VPC Network console accordingly.
3. To create the end-point to start the dask-scheduler, use `kubectl create -f dask-scheduler.yml`. This would open up a Jupyter notebook on port 30043 of your cluster, the token to enter the notebook is '*cuStreamz*'.
   Basically, you can run `kubectl get pods` to see that a new pod named `daskscheduler` has been created and is up and running. 
4. Now, similarly, create the Kafka and the cuStreamz-Dask client deployments using their respective .yml files. You can access the Jupyter endpoint of the cuStreamz-Dask client on port 30040. 
5. You can bash into the Kafka pod using `kubectl exec -it <name of Kafka pod> -- /bin/bash`. You will be able to see the Zookeeper and Kafka server that have been started using `screen -ls`. You can also list the contents of this deployment 
   and see the script `produce_random_to_kafka.py` script included. A topic named "cutreamz-topic" with 10 partitions has already been created as part of the deployment. Please feel free to change any configurations on the Kafka deployment as per your need. 

**Step 5: Start the pipeline end-to-end**
1. Run the cells in `dask_scheduler.ipynb` on the scheduler endpoint. This would use dask-kubernetes' KubeCluster to start a Dask cluster using the `dask_worker_spec.yml` with the diagnostics dashboard on port 30042.
2. If you face an authorization issue in `cluster.scale(1)`, you might require GKE admin privileges. Try `kubectl create clusterrolebinding serviceaccounts-cluster-admin --clusterrole=cluster-admin --group=system:serviceaccounts`
3. Now you have a Dask worker pod up and running. The worker spec. creates 8 Dask workers with 2 threads each. You can now scale up and down depending on the resources you have attached to your cluster.
4. Now run the `cuStreamz_job.ipynb` to start a cuStreamz job, and monitor the progess on the Dask dashboard. You might need to specify the Kafka pod's IP to consume data from the broker(s). Get the IP using `kubectl describe pod kafka_pod`.
5. You can push data into Kafka by bashing into the Kafka pod and then running the `produce_random_to_kafka.py` script.

**Step 6: You're streaming!**\
Please feel free to get in touch with us if you're facing any problems in the deployment process, and we'll try our best to look into them.
