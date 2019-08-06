FROM ubuntu:16.04

RUN ln -snf /bin/bash /bin/sh

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y emacs

# Adding wget and bzip2
RUN apt-get install -y wget bzip2

# Add sudo
RUN apt-get -y install sudo

#Installs screen
RUN apt-get -y install screen

#Installs vim-tiny
RUN apt-get -y install vim

#Installs OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

#Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

#Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

#Add user ubuntu with no password, add to sudo group
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu/
RUN chmod a+rwx /home/ubuntu/

#Installs miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /home/ubuntu/miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/home/ubuntu/miniconda/bin:${PATH}
RUN conda update -y conda

#Create cuStreamz environment
RUN conda create -n custreamz python=3.6
RUN echo "source activate custreamz" > /home/ubuntu/.bashrc
ENV PATH /home/ubuntu/miniconda/envs/custreamz/bin:$PATH

#Installs python-confluent-kafka, usjon, numpy
RUN source activate custreamz && \
    conda install -y -c conda-forge python-confluent-kafka && \
    conda install -y -c anaconda ujson && \
    conda install -y -c anaconda numpy

#Installs Kafka
RUN wget --quiet http://apache.claz.org/kafka/2.2.0/kafka_2.12-2.2.0.tgz
RUN tar -xzf kafka_2.12-2.2.0.tgz && rm kafka_2.12-2.2.0.tgz
ENV KAFKA_HOME /home/ubuntu/kafka_2.12-2.2.0

#Adds Shell script to start Zookeeper and Kafka server. Also adds Python script to produce random data to Kafka.
ADD . /home/ubuntu
USER root
RUN chmod +x ./start_kafka.sh && \
    chmod +x ./produce_random_to_kafka.py
USER ubuntu
