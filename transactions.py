import sys

from pyspark.mllib.fpm import FPGrowth
from pyspark import SparkConf, SparkContext


if __name__ == "__main__":
 	
    """
        Usage: transactions N
    """

    #Initialize Spark
    conf = SparkConf().setAppName("FPGrowth")
    sc = SparkContext(conf=conf)

    #Insert minimum number of transactions
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    #Load data and convert to RDD
    data = sc.textFile("hdfs://master:9000/user/user/input/")
    transactions = data.map(lambda line: line.strip().split(' '))
    
    #Find minSupport 
    m = transactions.count()
    x = n / m
    minS = round(x, 7)
    
    #Find frequent sets
    model = FPGrowth.train(transactions, minSupport=minS, numPartitions=10)
    result = model.freqItemsets().collect()
    
    #Save the output to a file    
    with open('frequent_sets.txt', 'w') as f:
        for re in result:
            f.write("{}\n".format(re))
    
    

