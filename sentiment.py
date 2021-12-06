from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)

stream = ssc.socketTextStream("localhost", 6100)

def model(df):
    data = df.select("feature0","feature1")
    split_data=data.randomSplit([0.8,0.2])
    train=split_data[0]
    
    test=split_data[1].withColumnRenamed("feature0","true_label")
    train_rows=train.count()
    test_rows=test.count()
    print("Total train :",train_rows)
    print("Total test :", test_rows)
    
    tokenizer = Tokenizer(inputCol="feature1", outputCol="SentimentWords")
    tokenizedTrain = tokenizer.transform(train)
    #tokenizedTrain.show(truncate=True, n=5)
    
    swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="MeaningfulWords")
    SwRemovedTrain = swr.transform(tokenizedTrain)
    #SwRemovedTrain.show(truncate=True, n=5)
    
    hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
    numericTrain = hashTF.transform(SwRemovedTrain).select('feature0', 'MeaningfulWords', 'features')
    #numericTrain.show(truncate=True, n=3)
    
    lr = LogisticRegression(labelCol="feature0", featuresCol="features", maxIter=10, regParam=0.01)
    model = lr.fit(numericTrain)
    print ("Training Done")
    
    tokenizedTest = tokenizer.transform(test)
    SwRemovedTest = swr.transform(tokenizedTest)
    numericTest = hashTF.transform(SwRemovedTest)
    #numericTest.show(truncate=True, n=2)
    
    raw_prediction = model.transform(numericTest)
    #raw_prediction.printSchema()
    
    Final_prediction = raw_prediction.select("MeaningfulWords", "prediction", "true_label")
    Final_prediction.show(n=4, truncate = False)
    
    Total_True=Final_prediction.filter(Final_prediction['prediction']==Final_prediction['true_label']).count()
    Alldata=Final_prediction.count()
    Accuracy=Total_True/Alldata
    print("Accuracy Score is:", Accuracy*100, '%')

def readMyStream(rdd):
    if not rdd.isEmpty():
        print("Batch:")
        data = rdd.collect()[0]
        nested_json_data = json.loads(data)
        json_data = nested_json_data.values()
        df = spark.createDataFrame(json_data)
        model(df)
        print("\n")
        #df.show()
        
stream.foreachRDD(lambda rdd: readMyStream(rdd))

ssc.start()
ssc.awaitTermination()