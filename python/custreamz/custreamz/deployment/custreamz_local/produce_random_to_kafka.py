#This script simulates a stream of data into a Kafka topic. Each Kafka message is a randomly generated 40-column JSON string.  

import confluent_kafka
from confluent_kafka import KafkaError, KafkaException
from confluent_kafka import TopicPartition
import logging
import sys
from time import sleep
import numpy as np
from random import randrange
import ujson
import math
import time

num_columns = 40

def column_names(size):
    base_cols = ["AppId{}", "PlayTime{}", "timestamp{}"]
    cols = []
    mult = math.ceil(size/len(base_cols))
    for i in range(mult):
        for c in base_cols:
            cols.append(c.format(i))
            if(len(cols) == size): break
    return cols

def generate_json(num_columns):
    dict_out = {}
    cols = column_names(num_columns)
    for col in cols:
        if col.startswith("AppId"): dict_out[col] = randrange(1,50000)
        elif col.startswith("PlayTime"): dict_out[col] = randrange(1,50000)
        else: dict_out[col] = randrange(1,50000)
    return ujson.dumps(dict_out)

# create logger
logger = logging.getLogger('client')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s'))
logger.addHandler(handler)

#create producer
producer_conf = {'bootstrap.servers': 'localhost:9092', 'compression.type':'snappy'}
producer = confluent_kafka.Producer(producer_conf)
topic0 = "custreamz-topic"

def delivery_callback(err, msg):
        if err:
            sys.stderr.write('%% Message failed delivery: %s\n' % err)
        else:
            sys.stderr.write('%% Message delivered to %s [%d] @ %o\n' %
                             (msg.topic(), msg.partition(), msg.offset()))

count = 0
start_time = int(round(time.time()))
try:
    while True:
        producer.produce(topic0, generate_json(num_columns), callback=delivery_callback)
        count = count + 1
        if count % 50000 == 0:
            producer.flush()
except KeyboardInterrupt:
    sys.stderr.write('%% Aborted by user\n')
end_time = int(round(time.time()))

print("%d %d" %(start_time, end_time))