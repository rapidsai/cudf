#!/bin/bash

screen -dmS zk bash -c "$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties"
screen -dmS ks bash -c "$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties"

sleep 2

$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 10 --topic custreamz-topic

/bin/bash
