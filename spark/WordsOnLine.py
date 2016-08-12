from __future__ import print_function

from pyspark import SparkContext


def max(a,b):
    if (a>b):
        return a
    else:
        return b


def main():
    """WordsOnLine"""
    
    
    fileBig = "/home/tcabel/Devel/test/big.txt"
    sc = SparkContext(appName="WordCount")
    textData = sc.textFile(fileBig).cache()

    maxWords = textData.map(lambda line: len(line.split())).reduce(max)
    outMaxWords = maxWords.collect()
    print("max words on a line : %i" % (outMaxWords))

    wordCounts = textData.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
    outWordCounts = wordCounts.take(20)
    for (word, count) in output:
        print("%s: %i" % (word, count))

    sc.stop()

if __name__ == "__main__": main()
