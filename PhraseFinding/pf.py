#!/usr/bin/env python
# encoding: utf-8

"""
@brief Phrase finding with spark
@param fg_year The year taken as foreground
@param f_unigrams The file containing unigrams
@param f_bigrams The file containing bigrams
@param f_stopwords The file containing stop words
@param w_info Weight of informativeness
@param w_phrase Weight of phraseness
@param n_workers Number of workers
@param n_outputs Number of top bigrams in the output
"""
from __future__ import division
import sys, math
from pyspark import SparkConf, SparkContext
from operator import add 

def process(line, fg_year, stopwords):
    tokens = line.split('\t')
    count = int(tokens[2])
    words = tokens[0].split()
    if not words[0] in stopwords and not words[1] in stopwords:
	if int(tokens[1]) == fg_year:
	    return (tokens[0], (count, 0))
	else:
	    return (tokens[0], (0, count))

def word_key(x):
    words = x[0].split()
    return (words[0], (words[1], x[1]))

def process2(line, fg_year, stopwords):
    tokens = line.split('\t')
    count = int(tokens[2])
    if not tokens[0] in stopwords:
	if (int(tokens[1])) == fg_year:
	   return (tokens[0], count)
        else:
           return (tokens[0], 0)

def zero(x):
    (w1, (w, n)) = x
    if not n:
        return (w[0], (w1, 0, w[1][0], w[1][1]))
    else:
        return (w[0], (w1, n, w[1][0], w[1][1]))

def zero2(x):
    (w2, (w, n)) = x
    if not n:
        return (w[0], w2, w[1], 0, w[2], w[3])
    else:
        return (w[0], w2, w[1], n, w[2], w[3])
def KL(p, q):
    return p * (math.log(p) - math.log(q))

def compute(x, w_info, w_phrase, uni_bigrams, uni_unigrams, bigrams_fg, bigrams_bg, unigrams_fg):
    (w1, w2, u1, u2, fg, bg) = x
    p1 = (fg + 1) / (uni_bigrams + bigrams_fg) 
    p2 = (bg + 1) / (uni_bigrams + bigrams_bg)
    p3 = (u1 + 1) / (uni_unigrams + unigrams_fg)
    p4 = (u2 + 1) / (uni_unigrams + unigrams_fg)
    
    p = KL(p1, p3*p4)
    i = KL(p1, p2)
    return (w1 +'-'+w2, w_phrase*p + w_info*i)

def  main(argv):
    # parse args
    fg_year = int(argv[1])
    f_unigrams = argv[2]
    f_bigrams = argv[3]
    f_stopwords = argv[4]
    w_info = float(argv[5])
    w_phrase = float(argv[6])
    n_workers = int(argv[7])
    n_outputs = int(argv[8])

    """ configure pyspark """
    conf = SparkConf().setMaster('local[{}]'.format(n_workers))  \
                      .setAppName(argv[0])
    sc = SparkContext(conf=conf)
    #global counter
    #counter = 0
    #sc.parallelize([1, 2, 3, 4, 5]).foreach(lambda x: f())

    # TODO: start your code here
    stopwords = sc.textFile(f_stopwords).collect()
    unigrams = sc.textFile(f_unigrams).collect()
	  
    # get bigrams RDD in form of (bigram, fgCount, bgCount)
    bigrams = sc.textFile(f_bigrams)
    bigrams2 = bigrams.map(lambda line: process(line, fg_year, stopwords)).filter(lambda x: x != None)
    bigrams3 = bigrams2.reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1]), numPartitions=n_workers).map(lambda x: word_key(x)).cache()
    uni_bigrams = bigrams3.count()
    (bigrams_fg, bigrams_bg) = bigrams3.map(lambda x: (x[1][1][0], x[1][1][1])).reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
  
    # get unigrams RDD in form of (unigram, fgCount, bgCount)
    unigrams = sc.textFile(f_unigrams)
    unigrams2 = unigrams.map(lambda line: process2(line, fg_year, stopwords)).filter(lambda x: x != None)
    unigrams3 = unigrams2.reduceByKey(add, numPartitions=n_workers).cache()
    uni_unigrams = unigrams3.count()
    unigrams_fg = unigrams3.map(lambda x: x[1]).reduce(add)
  
    # generate RDD (word1, word2, word1_fg, word2_fg, bigrams_fg, bigrams_bg)
    word1 = bigrams3.leftOuterJoin(unigrams3).map(lambda x: zero(x))
    word2 = word1.leftOuterJoin(unigrams3).map(lambda x: zero2(x))

    # compute probs
    probs = word2.map(lambda x: compute(x, w_info, w_phrase, uni_bigrams, uni_unigrams, bigrams_fg, bigrams_bg, unigrams_fg))
    scores = sorted(probs.collect(), key=lambda x:x[1], reverse=True)
    for score in scores[:n_outputs]:
    	print score[0] + ':' + str(score[1])
        
    """ terminate """
    sc.stop()



if __name__ == '__main__':
    main(sys.argv)

