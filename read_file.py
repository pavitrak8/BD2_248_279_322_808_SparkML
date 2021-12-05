from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
import json

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)

stream = ssc.socketTextStream("localhost", 6100)

def readMyStream(rdd):
    if not rdd.isEmpty():
        data = rdd.collect()[0]
        nested_json_data = json.loads(data)
        json_data = nested_json_data.values()
        df = spark.createDataFrame(json_data)
        df.show()
        
stream.foreachRDD(lambda rdd: readMyStream(rdd))

ssc.start()
ssc.awaitTermination()
