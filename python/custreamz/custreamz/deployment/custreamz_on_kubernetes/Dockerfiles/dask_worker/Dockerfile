#Installs Anaconda, cudf0.8, Dask by default
FROM rapidsai/rapidsai:cuda10.0-base-ubuntu16.04

#Installs python-confluent-kafka, ipywidgets, ipykernel
RUN source activate rapids && \
    conda update -n base -c defaults conda && \
    conda install -y -c anaconda ipywidgets && \
    conda install -y -c anaconda ipykernel && \
    pip install confluent-kafka

#Installs streamz
RUN source activate rapids && \
    git clone https://github.com/python-streamz/streamz.git && \
    cd ./streamz && \
    python setup.py build && \
    python setup.py install && \
    cd .. && \
    rm -rf streamz

ENTRYPOINT ["/bin/bash", "-c"]
