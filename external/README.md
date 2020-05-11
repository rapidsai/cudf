# External Datasources
cuDF uses the concept of ```datasources``` for implementations that load data and create dataframes. As the name
suggests ```external_datasources``` is an extension of this concept that allows for users to develop their own
```datasources``` implementation without having to directly modify the cuDF codebase.

External Datasources typically interact with complicated external systems that require more configuration
and oversight by the user. This abstraction allows for the implementor to handle this logic while at the same
time using the convenince of the cuDF ```datasource``` to handle parsing (JSON, CSV, Parquet, ORC, etc) and
creating the underlying cuDF Table instance.

External Datasources should be written as completely optional and cuDF does not depend on their existance to continue
its normal operation.

## Installing
Regardless of if you install an external datasource from Conda or from source you need to make sure the resulting shared object for 
the external datasource in a a location that cuDF knows about to load it. If you are using conda environments and install via conda 
you shouldn't have to worry about this as the shared objects should be installed in ```$CONDA_PREFIX/lib/external```. However it is 
good to keep in mind if you are having issues so you can check those files actually do exist there. If installing from source after building 
you can also install the shared objects in ```$CONDA_PREFIX/lib/external``` if you are not using conda or wish to place them somewhere 
else you can also set the environment variable ```EXTERNAL_DATASOURCE_LIB_PATH``` to the full path which contains your external datasource 
shared objects. This will be used by cuDF to located and load the shared objects at runtime.

### Conda
Installing with conda is straight forward. Since there are several datasources that can be installed it is best to reference the
[table](#external-datasources) listing the available datasources and their conda link.

### Source
While External Datasources will eventually be moved to its own repo for now it lives inside cuDF. For that reason we use the existing
```$CUDF_HOME/build.sh``` script to build the external datasources.

Note that while external datasources do not directly interact with cuDF you do need a build of cuDF that has the thin wrapper
to load the external datasources built. A build of libcudf that supports using external datasources can be made by running. 
```$CUDF_HOME/build.sh libcudf external``` this will ensure libcudf is built with external datasource support.

Building external datasources follows the same paradigm as cuDF and relies on conda environments as well. Once you have activated your ```cudf_dev```
environment ```conda activate cudf_dev``` you can simply run ```$CUDF_HOME/build.sh external``` to build all of the external sources and install
them in ```$CONDA_PREFIX/lib/external``` where they can be used.

## Using

### Using from Python
As compiled shared objects the external datasources can be used in numerous ways. However the two most common usages 
are listed out below.

#### Conjunction with cuDF
As long as libcudf was compiled with ```external``` datasource support there is nothing extra you need to do to use your
external datasources outside of the way cuDF works. In fact the only difference is an added function parameter on the 
Python readers that have been enabled to use external datasources specifying the datasource to load and use.

```python
import cudf

ex_ds_configs = {
    "metadata.broker.list": "localhost:9092",
    "topic": "libcudf-example"
}
ex_ds_id = "librdkafka-1.4.0" # Specifies we want to read from Kafka using the `librdkafka-1.4.0` datasource and expect CSV messages.
gdf = cudf.read_csv(ex_ds_id, ex_ds_configs)

# TODO: Note to reviewer this example is not actually wired up yet in this PR. Another PR will be opened to handle that.

```

So what just happened? Instead of suffering the overhead of
* Connecting to Kafka using a Python client.
* Reading messages with Python which in turn creates PyObjects and Strings.
* Perform unnecessary memory copies
* Pass the byte array to cudf.read_csv()

We allow cudf to handle all of that for us under the covers. In this manner we can ensure that the most efficient means
possible are used for us to create a cudf dataframe that has been sourced from Apache Kafka.

#### Ancillary Datasource Operations
External datasources are capable of more than just supplying bytes directly to cudf for Dataframe creation. It is 
also possible to define extra functions inside of external datasources that are not loaded by cuDF but can be loaded
by other libraries that may desire them. Custreamz is a good example of this. Custreamz interacts with Kafka heavily. 
As part of these interactions they need extra functionality to handle things like ```get_committed_offset()``` or ```get_watermark_offset()```. 
Those capabilities are clearly not things cuDF should handle however the external datasource still serves as a good place 
to keep this extra logic so that it can easily be used by libraries in the RapidsAI ecosystem that need them, like Custreamz. 
In the small example below we can see how Custreamz can use the external datasource logic to create both dataframes as well 
and use the extra functionality of the external datasource.

```python
from custreamz import kafka
from confluent_kafka import TopicPartition # Desired by CuStreamz but not required. 

kafka_configs = {
	"metadata.broker.list": "localhost:9092",
	"group.id": "custreamz-test",
}
topic = "example"
tps = []
tps.append(TopicPartition(topic, 0, 0)) # Set Topic, Partition, Offset (TOPPAR)

consumer = kafka.Consumer(kafka_configs) # Creates the instance of the external datasource
committed = consumer.committed(tps[0])
print("Committed Offsets: " + str(committed)) # List response for each topic partition

# Read the actual dataframe and uses the previously retrieved offset to set the start read position from the Kafka topic
gdf = consumer.read_gdf(topic=topic, partition=0, lines=True, start=committed[0], end=-1)
```

### Version Management
As long as the underlying library for the datasource supports it multiple versions of of an external datasource may be used.
This allows the freedom of interacting with several different versions of a system. An example might be interacting with
two different Kafka clusters each of which are different versions. This would allow a user to declare something like 
```librdkafka-0.9.0``` for an older cluster and ```librdkafka-1.4.0``` when dealing with a new cluster that does
not require the prescence of Apache Zookeeper. Example of how to do this can be seen in the [Using from Python](#using-from-python) section.

## External Datasources
External Datasources provided by the RapidsAI team.

| Unique ID        | Source           | Conda |
| ------------- |:-------------:|:----------:|
| librdkafka-1.4.0 | Apache Kafka Datasource - librdkafka version 1.4.0 | jdye64/cudf-kafka-datasource |
