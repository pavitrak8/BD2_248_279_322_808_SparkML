
#import the pyspark libraries
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
import json

from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.clustering import KMeans

#Calculation of performance metrics

def metric(metrics):
    cm=metrics.confusionMatrix().toArray() #Calculation of confusion matrix 
    precision=(cm[0][0])/(cm[0][0]+cm[1][0])   # Calculation of Precision metrics
    recall=(cm[0][0])/(cm[0][0]+cm[0][1])   #Calculation of Recall metrics 
    print("Confusion Matrix:")
    print(cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("\n")

# Models :
# 1.logistic regression model 
def logisticRegression(trainingData, testData):
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0) #Hyperparameters
    
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)
    
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    
   #Tuning hyperparameters    
    paramGrid = (ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.3, 0.5]).addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]).build())
 
    cv = CrossValidator(estimator=lr, estimatorParamMaps = paramGrid, evaluator=evaluator, numFolds=5) #K-fold cross validation
    cvModel = cv.fit(trainingData)
    predictions = cvModel.transform(testData)
    
    print("Logistic Regression")
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("Accuracy",(evaluator.evaluate(predictions))*100,"%")
    
    results = predictions.select(['prediction', 'label'])
    predictionAndLabels=results.rdd
    metrics = MulticlassMetrics(predictionAndLabels)
    
    #To calculate metrics
    metric(metrics)

# Naive Bayes model
def naiveBayes(trainingData, testData):
    nb = NaiveBayes(smoothing=1)
    
    model = nb.fit(trainingData)
    predictions = model.transform(testData)
	
    print("Naive Bayes:")    
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("Accuracy",(evaluator.evaluate(predictions))*100,"%")
    
    results = predictions.select(['prediction', 'label'])
    predictionAndLabels=results.rdd
    metrics = MulticlassMetrics(predictionAndLabels)
    
    #To calculate metrics
    metric(metrics)

#Random forest model
def randomForest(trainingData, testData):
    rf = RandomForestClassifier(labelCol='label',featuresCol='features',numTrees=100,maxDepth=4,maxBins=32) #The best parameters were selected 
    
    rfModel = rf.fit(trainingData)
    predictions = rfModel.transform(testData)
    
    print("Random Forest:")
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("Accuracy",(evaluator.evaluate(predictions))*100,"%")
    
    results = predictions.select(['prediction', 'label'])
    predictionAndLabels=results.rdd
    metrics = MulticlassMetrics(predictionAndLabels)

    #To calculate metrics
    metric(metrics)

# K-means clustering 
def kMeans(trainingData, testData):
    kmeans = KMeans().setK(3) #k=3 (could try with other optimal clusters as well)
    model = kmeans.fit(trainingData)

    #Make predictions
    predictions = model.transform(testData)

    print("K-Means Clustering:")
    #Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()
    dist = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(dist)) #gives the optimal clusters 
	
    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)


sc = SparkContext.getOrCreate()   #instantiating sparkcontext RDD
spark = SparkSession(sc) #Entry point to start programming with DataFrame and dataset.
ssc = StreamingContext(sc, 1) 

stream = ssc.socketTextStream("localhost", 6100) #connecting to localhost


#Preprocessing the data
def model(df):

    # regular expression tokenizer (splits the words as tokens and pattern can be used in text)
    regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="words")
    # stop words remover(removes unimportant words)
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
    # bag of words count
    countVectors = CountVectorizer(inputCol="filtered", outputCol="features", minDF=5)
    #String indexer -make the string column as indices (labels)
    label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")
    
    #Pipeline
    pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
    
    pipelineFit = pipeline.fit(df)
    dataset = pipelineFit.transform(df)
    #TrainTestSplit
    (train, test) = dataset.randomSplit([0.8, 0.2])
    
    logisticRegression(train, test) #LogisticRegression model 
    naiveBayes(train, test)  # NaiveBayes model
    randomForest(train, test) #randomForest model
    kMeans(train, test)   #clustering -> K-means clustering  
    
    
#Reading the streaming nested json and converting it to RDD    
def readMyStream(rdd):
    global i
    if not rdd.isEmpty():
        i+=1
        print("Batch",i,":")
        data = rdd.collect()[0]
        nested_json_data = json.loads(data)
        json_data = nested_json_data.values()
        df = spark.createDataFrame(json_data)  #converting RDD into dataframes 
        model(df)
        print("\n")
        #df.show()

i=0        
stream.foreachRDD(lambda rdd: readMyStream(rdd)) #streaming the data

ssc.start() #starting the streamingcontext
ssc.awaitTermination()#stopping the streamingcontext
