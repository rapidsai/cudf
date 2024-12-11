# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import confluent_kafka as ck
from cudf_kafka._lib.kafka import KafkaDatasource

import cudf


# Base class for anything class that needs to interact with Apache Kafka
class CudfKafkaClient:
    def __init__(self, kafka_configs):
        """
        Base object for any client that wants to interact with a Kafka broker.
        This object creates the underlying KafkaDatasource connection which
        is used to read data from Kafka and create cudf Dataframes.
        This class should not be directly instantiated.

        Parameters
        ----------
        kafka_configs : dict,
            Dict of Key/Value pairs of librdkafka
            configuration values. Full list of valid configuration
            options can be found at
            https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
        """

        self.kafka_configs = kafka_configs
        self.kafka_meta_client = KafkaDatasource(kafka_configs)
        self.ck_consumer = ck.Consumer(kafka_configs)

    def list_topics(self, specific_topic=None):
        """
        List the topics associated with the underlying Kafka Broker connection.

        Parameters
        ----------
        specific_topic : str,
            If specified this is the only topic that metadata information will
            be retrieved for. Otherwise metadata for all topics in the
            broker will be retrieved.
        """

        return self.kafka_meta_client.list_topics(
            b"" if specific_topic is None else specific_topic.encode()
        )

    def unsubscribe(self):
        """
        Stop all active consumption and remove consumer subscriptions
        to topic/partition instances
        """

        self.kafka_meta_client.unsubscribe()

    def close(self, timeout=10000):
        """
        Close the underlying socket connection to Kafka and
        clean up system resources
        """

        self.kafka_meta_client.close(timeout)


# Apache Kafka Consumer implementation
class Consumer(CudfKafkaClient):
    def __init__(self, kafka_configs):
        """
        Creates a KafkaConsumer object which allows for all valid Kafka
        consumer type operations such as reading messages, committing
        offsets, and retrieving the current consumption offsets.

        Parameters
        ----------
        kafka_configs : dict,
            Dict of Key/Value pairs of librdkafka
            configuration values. Full list of valid configuration
            options can be found at
            https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
        """

        super().__init__(kafka_configs)

    def read_gdf(
        self,
        topic=None,
        partition=0,
        lines=True,
        start=0,
        end=0,
        batch_timeout=10000,
        delimiter="\n",
        message_format="json",
    ):
        r"""
        Read messages from the underlying KafkaDatasource connection and create
        a cudf Dataframe

        Parameters
        ----------
        topic : str,
            Name of the Kafka topic that the messages
            should be read from
        partition : int,
            Partition number on the specified topic that
            should be read from
        lines : {{ True, False }}, default True,
            Whether messages should be treated as individual lines
        start : int, default 0,
            The beginning offset that should be used when
            reading a range of messages
        end : int, default 0,
            The last offset that will be read when
            reading a range of messages
        batch_timeout : int, default 10000,
            Amount of time to wait on the
            reading of the messages from Kafka in Milliseconds
        delimiter : str, default "\n",
            If lines=True this is the delimiter that
            will be placed between all messages that are read from Kafka
        message_format : {{ 'avro', 'csv', 'json', 'orc', 'parquet' }},
        default 'json',
            Format of the messages that will be read from Kafka.
            This dictates which underlying cudf reader will be invoked the
            create the Dataframe.

        Returns
        -------
        DataFrame
        """

        if topic is None:
            raise ValueError(
                "ERROR: You must specify the topic "
                "that you want to consume from"
            )

        kafka_datasource = KafkaDatasource(
            self.kafka_configs,
            topic.encode(),
            partition,
            start,
            end,
            batch_timeout,
            delimiter.encode(),
        )

        cudf_readers = {
            "json": cudf.io.read_json,
            "csv": cudf.io.read_csv,
            "orc": cudf.io.read_orc,
            "avro": cudf.io.read_avro,
            "parquet": cudf.io.read_parquet,
        }

        if message_format == "json":
            result = cudf_readers[message_format](
                kafka_datasource, engine="cudf", lines=True
            )
        else:
            result = cudf_readers[message_format](
                kafka_datasource, engine="cudf"
            )

        # Close up the cudf datasource instance
        # TODO: Ideally the C++ destructor should handle the
        # unsubscribe and closing the socket connection.
        kafka_datasource.unsubscribe()
        kafka_datasource.close(batch_timeout)

        if result is not None:
            if isinstance(result, cudf.DataFrame):
                return result
            else:
                return cudf.DataFrame._from_data(result)
        else:
            # empty Dataframe
            return cudf.DataFrame()

    def committed(self, partitions, timeout=10000):
        """
        Retrieves the last successfully committed Kafka offset of the
        underlying KafkaDatasource connection.

        Parameters
        ----------
        partitions : list,
            Topic/Partition instances that specify the TOPPAR
            instances the offsets should be retrieved for
        timeout : int, default 10000,
            Max time to wait on the response from
            the Kafka broker in milliseconds

        Returns
        -------
        tuple
            Tuple of ck.TopicPartition objects
        """

        toppars = [
            ck.TopicPartition(
                part.topic,
                part.partition,
                self.kafka_meta_client.get_committed_offset(
                    part.topic.encode(), part.partition
                ),
            )
            for part in partitions
        ]

        return toppars

    def get_watermark_offsets(self, partition, timeout=10000, cached=False):
        """
        Retrieve the low and high watermark offsets from the Kafka consumer

        Returns
        -------
        Tuple with a [low, high] value

        Examples
        --------
        >>> from custream import kafka
        >>> kafka_configs = {
        ... "metadata.broker.list": "localhost:9092",
        ... "enable.partition.eof": "true",
        ... "group.id": "groupid",
        ... "auto.offset.reset": "earliest",
        ... "enable.auto.commit": "false"
        ... }
        >>> consumer = kafka.KafkaHandle(kafka_configs,
        ... topics=["kafka-topic"], partitions=[0]))
        >>> low, high = consumer.get_watermark_offsets("kafka-topic", 0)
        """

        offsets = ()

        try:
            offsets = self.kafka_meta_client.get_watermark_offset(
                topic=partition.topic.encode(),
                partition=partition.partition,
                timeout=timeout,
                cached=cached,
            )
        except RuntimeError:
            raise RuntimeError("Unable to connect to Kafka broker")

        if len(offsets) != 2:
            raise RuntimeError(
                f"Multiple watermark offsets encountered. "
                f"Only 2 were expected and {len(offsets)} encountered"
            )

        if offsets[b"low"] < 0:
            offsets[b"low"] = 0

        if offsets[b"high"] < 0:
            offsets[b"high"] = 0

        return offsets[b"low"], offsets[b"high"]

    def commit(self, offsets=None, asynchronous=True):
        """
        Takes a list of ck.TopicPartition objects and commits their
        offset values to the KafkaDatasource connection

        Parameters
        ----------
        offsets : list,
            ck.TopicPartition objects containing the
            Topic/Partition/Offset values to be committed to the Kafka broker
        asynchronous : {{ True, False }}, default True,
            True to wait on
            Kafka broker response to commit request and False otherwise
        """

        for offs in offsets:
            self.kafka_meta_client.commit_offset(
                offs.topic.encode(), offs.partition, offs.offset
            )

    def poll(self, timeout=None):
        """
        Consumes a single message, calls callbacks and returns events.

        The application must check the returned Message object's
        Message.error() method to distinguish between proper messages
        (error() returns None), or an event or error
        (see error().code() for specifics).

        Parameters
        ----------
        timeout : float
            Maximum time to block waiting for message, event or callback
            (default: infinite (None translated into -1 in the
            library)). (Seconds)
        """
        return self.ck_consumer.poll(timeout)
