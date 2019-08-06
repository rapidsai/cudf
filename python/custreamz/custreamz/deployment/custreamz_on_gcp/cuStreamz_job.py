from distributed import Client
#This creates a Dask client that submits jobs to the Dask scheduler that then assigns tasks to the workers.  
client = Client("localhost:8786")
client.get_versions(check=True)
client

###### Please write your cuStreamz job below ####

from streamz import Stream
import numpy as np
from streamz.dataframe import DataFrame
import time
import math
import confluent_kafka
import json
import os
import cudf
    
#Simple cuStreamz function
def gpu_preprocess_simple_agg(messages):
    preprocess_start_time = int(round(time.time()))
    size = len(messages)*len(messages[0])    
    json_input_string = "\n".join([msg.decode('utf-8') for msg in messages])
    pre_gpu_timestamp = int(round(time.time()))
    gdf = cudf.read_json(json_input_string, lines=True, engine='cudf') #Converts JSON-encoded string messages from Kafka into cudf DataFrame.
    preprocess_end_time = int(round(time.time()))
    
    #Simple aggregations
    gdf['Count'] = 1
    num_rows = gdf['Count'].sum()
    gdf1 = gdf.groupby(['AppId0','PlayTime0']).mean()
    
    agg_end_time = int(round(time.time()))
    return "{0},{1},{2},{3},{4},{5}".format(num_rows, preprocess_start_time, pre_gpu_timestamp, preprocess_end_time, agg_end_time, size)

#Kafka topic to read from, has 24 partitions in this example
topic = "custreamz-topic"

#Kafka brokers 
bootstrap_servers = 'localhost:9092'

#Kafka consumer configuration
consumer_conf = {'bootstrap.servers': bootstrap_servers, 'group.id': 'custreamz', 'session.timeout.ms': 60000}

#Polling Kafka every 10s and getting the messages as a batch using Dask workers
stream = Stream.from_kafka_batched(topic, consumer_conf, poll_interval='10s', npartitions=24, asynchronous = True, dask= True)

#Applying the function gpu_preprocess_simple_agg on every batch polled from Kafka, and sinking output into a list
final_output = stream.map(gpu_preprocess_simple_agg).buffer(10000).gather().sink_to_list()

#Starting the stream
stream.start()

final_output
