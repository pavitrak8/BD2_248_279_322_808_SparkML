from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext('local',"Project")
ssc = StreamingContext(sc, 1)

file = ssc.socketTextStream("localhost", 6100)
#data = file.flatMap(lambda line: line.split(","))
#data.pprint()

ssc.start()
ssc.awaitTermination()
