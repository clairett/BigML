run:
	time spark-submit --driver-memory 2G pf.py 1960 data/unigram_war.txt data/bigram_war.txt data/stopwords.txt 100 1 1 10 > result.txt 2>result.log

run2:
	time spark-submit --driver-memory 2G pf.py 1960 /afs/cs.cmu.edu/project/bigML/phrases/subsample_unigram /afs/cs.cmu.edu/project/bigML/phrases/subsample_bigram /afs/cs.cmu.edu/project/bigML/phrases/stopword.txt 100 1 16 10 > result.txt 2>result.log

war1960:
	spark-submit --driver-memory 2G pf.py 1960 data/unigram_war.txt data/bigram_war.txt data/stopwords.txt 100 50 5 50 > war1960.txt
	python gen_wc.py war1960.txt war1960.png

war1990:
	spark-submit --driver-memory 2G pf.py 1990 data/unigram_war.txt data/bigram_war.txt data/stopwords.txt 100 1 5 50 > war1990.txt
	python gen_wc.py war1990.txt war1991.png

apple1980:
	spark-submit --driver-memory 2G pf.py 1980 data/unigram_apple.txt data/bigram_apple.txt data/stopwords.txt 100 50 5 50 > apple1980.txt
	python gen_wc.py apple1980.txt apple1980.png

apple1990:
	spark-submit --driver-memory 2G pf.py 1990 data/unigram_apple.txt data/bigram_apple.txt data/stopwords.txt 100 1 5 50 > apple1990.txt
	python gen_wc.py apple1990.txt apple1991.png

apple1970:
	spark-submit --driver-memory 2G pf.py 1970 data/unigram_apple.txt data/bigram_apple.txt data/stopwords.txt 100 50 5 50 > apple1970.txt
	python gen_wc.py apple1970.txt apple1970.png

apple1960:
	spark-submit --driver-memory 2G pf.py 1960 data/unigram_apple.txt data/bigram_apple.txt data/stopwords.txt 100 50 5 50 > apple1960.txt
	python gen_wc.py apple1970.txt apple1960.png
handin:
	tar cvf handin.tar result.txt result.log pf.py
