spark

## config


## first app
### python
First, we need to import spark and initialize a *Spark Context*

{% highlight python %}
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
{% endhighlight %}

### launch
to launch a script, use *spark-submit* for example:
 - **local launch**
{% highlight shell %}
 /home/tcabel/Devel/test/spark-1.6.1-bin-hadoop2.4/bin/spark-submit --master local[4] --conf spark.executor.memory=4g --conf spark.eventLog.compress=true WordsOnLine.py --driver-java-options=-Dlog4j.logLevel=info
{% endhighlight %}
_Note : we can also use local[*] to use all the logical cores_

 - **Run on a Spark standalone cluster in client deploy mode** 
{% highlight shell %}
./bin/spark-submit --class org.apache.spark.examples.SparkPi --master spark://207.184.161.138:7077 --executor-memory 20G --total-executor-cores 100 /path/to/examples.jar 1000
{% endhighlight %}

 - **Run on a Spark standalone cluster in cluster deploy mode with supervise**
{% highlight shell %}
./bin/spark-submit --class org.apache.spark.examples.SparkPi --master spark://207.184.161.138:7077 --deploy-mode cluster --supervise --executor-memory 20G --total-executor-cores 100 /path/to/examples.jar 1000
{% endhighlight %}

 - **to launch with jupyter**
add a file ~/.ipython/profile_default/startup/00-pyspark-steup.py
import os
spark_home = '/home/tcabel/Devel/test/spark-1.6.1-bin-hadoop2.4'
sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.9-src.zip'))
execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))

then 
export SPARK_HOME=/home/tcabel/Devel/test/spark-1.6.1-bin-hadoop2.4/
export PYSPARK_SUBMIT_ARGS="--master local[*] --conf spark.executor.memory=4g --conf spark.eventLog.compress=true"
IPYTHON_OPTS="notebook" $SPARK_HOME/bin/pyspark

## Resilient Distributed Datasets (RDDs)

Spark revolves around the concept of a resilient distributed dataset (RDD), which is a fault-tolerant collection of elements that can be operated on in parallel. There are two ways to create RDDs: parallelizing an existing collection in your driver program, or referencing a dataset in an external storage system, such as a shared filesystem, HDFS, HBase, or any data source offering a Hadoop InputFormat. Parallelized collections are created by calling SparkContext’s parallelize method on an existing iterable or collection in your driver program.

{% highlight python %}
data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)
{% endhighlight %}

Text file RDDs can be created using SparkContext’s textFile method. This method takes an URI for the file (either a local path on the machine, or a hdfs://, s3n://, etc URI) and reads it as a collection of lines. Here is an example invocation:

{% highlight python %}
distFile = sc.textFile("data.txt")
{% endhighlight %}

## RDD Operations

### Basics

{% highlight python %}
lines = sc.textFile("data.txt")
lineLengths = lines.map(lambda s: len(s))
totalLength = lineLengths.reduce(lambda a, b: a + b)
{% endhighlight %}

### Understanding closures
One of the harder things about Spark is understanding the scope and life cycle of variables and methods when executing code across a cluster. RDD operations that modify variables outside of their scope can be a frequent source of confusion. In the example below we’ll look at code that uses *foreach()* to increment a counter, but similar issues can occur for other operations as well.
Consider the naive RDD element sum below, which may behave differently depending on whether execution is happening within the same JVM. A common example of this is when running Spark in local mode (--master = local[n]) versus deploying a Spark application to a cluster (e.g. via spark-submit to YARN):

{% highlight python %}
counter = 0
rdd = sc.parallelize(data)

# Wrong: Don't do this!!
def increment_counter(x):
    global counter
    counter += x
rdd.foreach(increment_counter)

print("Counter value: ", counter)
{% endhighlight %}

To ensure well-defined behavior in these sorts of scenarios one should use an **Accumulator**. Accumulators in Spark are used specifically to provide a mechanism for safely updating a variable when execution is split up across worker nodes in a cluster. The Accumulators section of this guide discusses these in more detail.

#### Printing elements of an RDD

Another common idiom is attempting to print out the elements of an RDD using `rdd.foreach(println)` or `rdd.map(println)`. On a single machine, this will generate the expected output and print all the RDD’s elements. However, in cluster mode, the output to stdout being called by the executors is now writing to the executor’s stdout instead, not the one on the driver, so stdout on the driver won’t show these! To print all elements on the driver, one can use the `collect()` method to first bring the RDD to the driver node thus: `rdd.collect().foreach(println)`. This can cause the driver to run out of memory, though, because collect() fetches the entire RDD to a single machine; if you only need to print a few elements of the RDD, a safer approach is to use the `take()`: `rdd.take(100).foreach(println)`.


### Working with Key-Value Pairs
While most Spark operations work on RDDs containing any type of objects, a few special operations are only available on RDDs of key-value pairs. The most common ones are distributed “shuffle” operations, such as grouping or aggregating the elements by a key.

In Python, these operations work on RDDs containing built-in Python tuples such as (1, 2). Simply create such tuples and then call your desired operation.

For example, the following code uses the reduceByKey operation on key-value pairs to count how many times each line of text occurs in a file:

{% highlight python %}
lines = sc.textFile("data.txt")
pairs = lines.map(lambda s: (s, 1))
counts = pairs.reduceByKey(lambda a, b: a + b)
{% endhighlight %}


### Transformations
The following table lists some of the common transformations supported by Spark. Refer to the RDD API doc (Scala, Java, Python, R) and pair RDD functions doc (Scala, Java) for details.
 -**map(func)** Return a new distributed dataset formed by passing each element of the source through a function func.
 -**filter(func)** Return a new dataset formed by selecting those elements of the source on which func returns true.
 -**flatMap(func)** Similar to map, but each input item can be mapped to 0 or more output items (so func should return a Seq rather than a single item). 
 -**union(otherDataset)** Return a new dataset that contains the union of the elements in the source dataset and the argument.
 -**intersection(otherDataset)** Return a new RDD that contains the intersection of elements in the source dataset and the argument.
 -**distinct([numTasks]))** Return a new dataset that contains the distinct elements of the source dataset.
 -**groupByKey([numTasks])** When called on a dataset of (K, V) pairs, returns a dataset of (K, Iterable<V>) pairs.
 -**reduceByKey(func, [numTasks])** When called on a dataset of (K, V) pairs, returns a dataset of (K, V) pairs where the values for each key are aggregated using the given reduce function func, which must be of type (V,V) => V. Like in groupByKey, the number of reduce tasks is configurable through an optional second argument. 

_Note: If you are grouping in order to perform an aggregation (such as a sum or average) over each key, using reduceByKey or aggregateByKey will yield much better performance._
_Note: By default, the level of parallelism in the output depends on the number of partitions of the parent RDD. You can pass an optional numTasks argument to set a different number of tasks._


### Actions 
 -**reduce(func)** Aggregate the elements of the dataset using a function func (which takes two arguments and returns one). The function should be commutative and associative so that it can be computed correctly in parallel.
 - **collect()** Return all the elements of the dataset as an array at the driver program. This is usually useful after a filter or other operation that returns a sufficiently small subset of the data.
 - **count()** Return the number of elements in the dataset.
 - **first()** Return the first element of the dataset (similar to take(1)).
 - **take(n)** Return an array with the first n elements of the dataset. 
 - **foreach(func)** Run a function func on each element of the dataset. This is usually done for side effects such as updating an Accumulator or interacting with external storage systems.
_Note: modifying variables other than Accumulators outside of the foreach() may result in undefined behavior. See Understanding closures for more details._

You can chain operations together, but keep in mind that the computation only runs when you call an action.

N.B: all operations made on RDDs are registered in a **DAG (direct acyclic graph)**: this is the lineage principle. In this way if a partition is lost, Spark can rebuilt automatically the partition thanks to this DAG.


### RDD Persistence
One of the most important capabilities in Spark is persisting ( *persist()* or *cache* ) a dataset in memory across operations. When you persist an RDD, each node stores any partitions of it that it computes in memory and reuses them in other actions on that dataset (or datasets derived from it). 

In addition, each persisted RDD can be stored using a different storage level, allowing you, for example, to persist the dataset on disk, persist it in memory but as serialized Java objects (to save space), replicate it across nodes, or store it off-heap in Tachyon. These levels are set by passing a StorageLevel object (Scala, Java, Python) to persist(). The cache() method is a shorthand for using the default storage level, which is StorageLevel.MEMORY_ONLY
The full set of storage levels is:
 - **MEMORY_ONLY** Store RDD as deserialized Java objects in the JVM. If the RDD does not fit in memory, some partitions will not be cached and will be recomputed on the fly each time they're needed. 
 - **MEMORY_AND_DISK** Store RDD as deserialized Java objects in the JVM. If the RDD does not fit in memory, store the partitions that don't fit on disk, and read them from there when they're needed. 
 - **MEMORY_ONLY_SER** Store RDD as serialized Java objects (one byte array per partition). This is generally more space-efficient than deserialized objects, especially when using a fast serializer, but more CPU-intensive to read. 
 - **MEMORY_AND_DISK_SER** 
 - **DISK_ONLY**

### Shared Variable
#### Broadcast
Broadcast variables allow the programmer to keep a read-only variable cached on each machine rather than shipping a copy of it with tasks. They can be used, for example, to give every node a copy of a large input dataset in an efficient manner. Spark also attempts to distribute broadcast variables using efficient broadcast algorithms to reduce communication cost.

{% highlight python %}
broadcastVar = sc.broadcast([1, 2, 3])
broadcastVar.value
{% endhighlight %}

After the broadcast variable is created, it should be used instead of the value v in any functions run on the cluster so that v is not shipped to the nodes more than once. In addition, the object v should not be modified after it is broadcast in order to ensure that all nodes get the same value of the broadcast variable (e.g. if the variable is shipped to a new node later).


#### Accumulators
Accumulators are variables that are only “added” to through an associative operation and can therefore be efficiently supported in parallel. They can be used to implement counters (as in MapReduce) or sums. Spark natively supports accumulators of numeric types, and programmers can add support for new types. If accumulators are created with a name, they will be displayed in Spark’s UI. This can be useful for understanding the progress of running stages (NOTE: this is not yet supported in Python).

An accumulator is created from an initial value v by calling SparkContext.accumulator(v). Tasks running on the cluster can then add to it using the add method or the += operator (in Scala and Python). However, they cannot read its value. Only the driver program can read the accumulator’s value, using its value method.

{% highlight python %}
accum = sc.accumulator(0)
sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
accum.value
{% endhighlight %}

While this code used the built-in support for accumulators of type Int, programmers can also create their own types by subclassing AccumulatorParam. The AccumulatorParam interface has two methods: zero for providing a “zero value” for your data type, and addInPlace for adding two values together. For example, supposing we had a Vector class representing mathematical vectors, we could write:

{% highlight python %}
class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return Vector.zeros(initialValue.size)

    def addInPlace(self, v1, v2):
        v1 += v2
        return v1

# Then, create an Accumulator of this type:
vecAccum = sc.accumulator(Vector(...), VectorAccumulatorParam())
{% endhighlight %}


For accumulator updates performed inside actions only, Spark guarantees that each task’s update to the accumulator will only be applied once, i.e. restarted tasks will not update the value. In transformations, users should be aware of that each task’s update may be applied more than once if tasks or job stages are re-executed.

Accumulators do not change the lazy evaluation model of Spark. If they are being updated within an operation on an RDD, their value is only updated once that RDD is computed as part of an action. Consequently, accumulator updates are not guaranteed to be executed when made within a lazy transformation like map(). The below code fragment demonstrates this property:

{% highlight python %}
accum = sc.accumulator(0)
def g(x):
  accum.add(x)
  return f(x)
data.map(g)
# Here, accum is still 0 because no actions have caused the `map` to be computed.
{% endhighlight %}




source : http://spark.apache.org/docs/latest/programming-guide.html


