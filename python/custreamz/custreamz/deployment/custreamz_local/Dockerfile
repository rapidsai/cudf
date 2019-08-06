#Installs Anaconda3, cudf0.8, Dask by default
FROM rapidsai/rapidsai:cuda10.0-base-ubuntu16.04

#Installs python-confluent-kafka, ipywidgets, ipykernel
RUN source activate rapids && \
    conda update -n base -c defaults conda && \
    conda install -y -c anaconda ipywidgets && \
    conda install -y -c anaconda ipykernel && \
    conda install -c conda-forge python-confluent-kafka 

#Installs streamz
RUN source activate rapids && \
    git clone https://github.com/python-streamz/streamz.git && \
    cd ./streamz && \
    python setup.py build && \
    python setup.py install && \
    cd .. && \
    rm -rf streamz

ADD . /rapids

#Add kernel to Jupyter Notebook
RUN source activate rapids && \
    python -m ipykernel install --user --name rapids --display-name "custreamz-dask_client"

ENTRYPOINT ["/bin/bash", "-c"]

#Start Jupyter Notebook on port 8888
CMD ["source activate rapids && jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token='cuStreamz'"]
