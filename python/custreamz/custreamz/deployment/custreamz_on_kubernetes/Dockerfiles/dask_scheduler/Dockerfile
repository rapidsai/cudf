#Installs Anaconda, cudf0.8, Dask by default
FROM rapidsai/rapidsai:cuda10.0-base-ubuntu16.04

#Installs python-confluent-kafka, ipywidgets, ipykernel
RUN source activate rapids && \
    conda update -n base -c defaults conda && \
    conda install -y -c anaconda ipywidgets && \
    conda install -y -c anaconda ipykernel && \
    pip install confluent-kafka

#Add kernel to Jupyter Notebook
RUN source activate rapids && \
    python -m ipykernel install --user --name rapids --display-name "custreamz-dask_scheduler"

#Install Dask Kubernetes
RUN source activate rapids && \
    conda install -y -c conda-forge dask-kubernetes

ENTRYPOINT ["/bin/bash", "-c"]

ADD . /rapids

#Start Jupyter Notebook on port 8888
CMD ["source activate rapids && jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token='cuStreamz'"]
