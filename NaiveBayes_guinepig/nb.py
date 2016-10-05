from __future__ import division
from guineapig import *
import re, math


# supporting routines can go here
def each_label(labels):
    for label in labels:
        yield label


def each_word((labels, doc)):
    for label in labels:
        for word in doc:
            word = re.sub('\W', '', word)
            if len(word) > 0:
                yield (label, word)


def eachTerm((docid, doc)):
    for term in doc:
        term = re.sub('\W', '', term)
        if len(term) > 0:
            yield (docid, term)

def tokens(doc):
    for term in doc:
        term = re.sub('\W', '', term)
        if len(term) > 0:
            yield term


def addToDict(accum, (label, count)):
    accum[label] = count
    return accum


def update_dict(accum, d):
    for key, val in d.iteritems():
        accum[key] = accum.get(key, 0) + val
    return accum


def compute_prior(d):
    for key, val in d.iteritems():
        d[key] = math.log((val+1)/(sum(d.values())+len(d)))
    return d


def get_max_prob(accum, (d, d1, count, d2)):
    for key, val in d2.iteritems():
        numerator = d.get(key, 0) + 1
        accum[key] = accum.get(key, val) + math.log(numerator/(d1[key]+count))
    return accum


def predict_label((docid, d)):
    max_prob = -float("inf")
    max_label = ""
    for key, val in d.iteritems():
        if val > max_prob:
            max_label = key
            max_prob = val
    return (docid, max_label, max_prob)


def get_word_label(accum, d):
    for key, val in d.iteritems():
        accum[key] = accum.get(key, 0)+val
    return accum


def calc_accuracy(accum, x):
    (id, label, labels, docs_count) = x
    if label in labels.split(','):
        accum += 1/docs_count
    return accum

# always subclass Planner
class NB(Planner):
    # params is a dictionary of params given on the command line.
    # e.g. trainFile = params['trainFile']
    params = GPig.getArgvParams()
    # parse the data into (docid, label, doc)
    fields = ReadLines(params['trainFile']) | ReplaceEach(by=lambda line: line.strip().split("\t"))

    # (word, {label:count})
    train = ReplaceEach(fields, by=lambda (docid, label, doc): (label.split(","), doc.split())) | Flatten(by=each_word) \
        | Group(by=lambda (label, word): (label, word), retaining=lambda x:1, reducingTo=ReduceToSum()) \
        | Group(by=lambda (pair, count):pair[1], retaining=lambda (pair, count):(pair[0], count), reducingTo=ReduceTo(dict, by=lambda accum, (label, count):addToDict(accum, (label, count))))

    # doc label (label, count)
    prior = ReplaceEach(fields, by=lambda (docid, label, doc): label.split(",")) | Flatten(by=each_label) | Group(by=lambda x: x, retaining=lambda x:1, reducingTo=ReduceToSum()) \
        | Group(by=lambda x:"Prob", reducingTo=ReduceTo(dict, by=lambda accum, (label, count):addToDict(accum, (label, count)))) \
        | ReplaceEach(by=lambda x:compute_prior(x[1]))

    num_word = train | Distinct() | Group(by=lambda x:"Word", reducingTo=ReduceToCount())
    word_label = Group(train, by=lambda x:1, retaining=lambda (word, d): d, reducingTo=ReduceTo(dict, by=lambda accum, d:get_word_label(accum, d)))

    # load test request data
    testFields = ReadLines(params['testFile']) | ReplaceEach(by=lambda line: line.strip().split("\t"))
    # (docid, word)
    testData = ReplaceEach(testFields, by=lambda (docid,label,doc): (docid, doc.split())) | Flatten(by=eachTerm)
    request = Join( Jin(testData, by=lambda (docid, word):word), Jin(train, by=lambda (word, d):word)) | ReplaceEach(by=lambda ((docid, word), (word_,d)):(docid, word, d)) \
            | Augment(sideviews=[word_label, num_word, prior], loadedBy=lambda v1, v2, v3: [GPig.onlyRowOf(v1), GPig.onlyRowOf(v2), GPig.onlyRowOf(v3)])| ReplaceEach(by=lambda ((docid, word, d), ((dummy, d1), (dummy2, count),d2)):(docid, word, d, d1, count, d2))

    output = Group(request, by=lambda (docid, w, d, d1, count, d2):docid, retaining=lambda (docid, w, d, d1, count, d2): (d, d1, count, d2), reducingTo=ReduceTo(dict, by=lambda accum, (d, d1, count, d2):get_max_prob(accum,(d, d1, count, d2)))) \
            | ReplaceEach(by=lambda (docid, d):predict_label((docid, d)))

    # Test Accuracy
    # gold = Map(testFields, by=lambda (id,labels,doc): (id, labels))
    # compare = Join(Jin(output, by=lambda (id, label, max_prob):id), Jin(gold, by=lambda (id,labels):id)) | Map(by=lambda (((id1, label, max_prob)),(id2,labels)): (id1, label, labels))
    # test_docs_num = Group(ReadLines(params['testFile']), by=lambda x:'test_docs_count', reducingTo=ReduceToCount())
    # accuracy = Augment(compare, sideview=test_docs_num, loadedBy=lambda x:GPig.onlyRowOf(x)) | Map(by=lambda ((id,label,labels),(_, docs_count)): (id, label, labels, docs_count)) \
    # 				| Group(by=lambda x:'Accuracy', reducingTo=ReduceTo(float, by=lambda accum, x: calc_accuracy(accum,x)))


# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here
