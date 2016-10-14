javac LR.java 
for((i=1;i<=20;i++));
do gshuf abstract.small.train
done | java -Xmx2580m LR 1000000 0.5 1 20 44925 abstract.small.test > log2
