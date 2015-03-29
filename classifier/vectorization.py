__author__ = 'shaughnfinnerty'
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.stem.snowball import EnglishStemmer
import string
import numpy
import arff
from sklearn.naive_bayes import GaussianNB
import codecs
import itertools
import pickle

from sklearn import svm
def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf8_data, delimiter='\t', dialect=dialect,  **kwargs)
    for row in csv_reader:
        # yield [unicode(cell, 'utf-8') for cell in row]
        ids.append(unicode(row[0],'utf-8'))
        polarity.append(unicode(row[2],'utf-8'))
        text.append(unicode(row[3],'utf-8'))
        # print(row[3])


filename = 'semeval_twitter_data.txt'

ids = []
polarity = []
text = []
reader = unicode_csv_reader(open(filename))

print len(text)

stemmer = EnglishStemmer();
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

base_vectorizer = CountVectorizer(stop_words='english', min_df=1)
p = base_vectorizer.build_preprocessor();
t = base_vectorizer.build_tokenizer();

def stemming_analyzer(document):
    stem_it = stemmer.stem
    return map(stem_it, t(p(document)))

vectorizer = CountVectorizer(stop_words='english', min_df=4, analyzer=stemming_analyzer)
# vectorizer = CountVectorizer(stop_words='english', min_df=10)

# vectorizer.get_feature_names()



features = vectorizer.fit_transform(text)
print "Features Created: " + str(len(vectorizer.get_feature_names()));

# tfidf_transformer = TfidfTransformer()
# features = tfidf_transformer.fit_transform(features)

# print X_train_tfidf.shape
# print type(X_train_tfidf)
# print X_train_tfidf[2]
# print features[1]
# t = vectorizer.transform(text[:1000])
# t_tf_idf = tfidf_transformer.transform(t)
# # print vectorizer.transform("I hate this!")
#
# clf = svm.SVC(kernel='linear')
# # clf = GaussianNB()
# clf.fit(features, polarity)
# print("classifier trained")
# print clf.predict(t)

# ef using_tocoo_izip(x):
#     cx = x.tocoo()
#     for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
#         (i,j,v)

def to_sparse_arff(file):
    file_analytics = open("msg_analytics_pos-neg-obj-scores-normalized.obj")
    msg_analytics = pickle.load(file_analytics)
    file_analytics.close()
    print("Length of msg analytics: " + str(len(msg_analytics)));

    with codecs.open(file, "w", "utf-8") as f:
        f.write("@RELATION opinion\n")
        length = len(vectorizer.get_feature_names())
        for name in vectorizer.get_feature_names():
            f.write("@ATTRIBUTE " + name + " numeric\n");
        f.write("@ATTRIBUTE smileyfaces numeric\n");
        f.write("@ATTRIBUTE sadfaces numeric\n");
        f.write("@ATTRIBUTE exclamationmarks numeric\n");
        f.write("@ATTRIBUTE questionmarks numeric\n");
        f.write("@ATTRIBUTE posscore numeric\n");
        f.write("@ATTRIBUTE negscore numeric\n");
        f.write("@ATTRIBUTE objscore numeric\n");

        f.write("@ATTRIBUTE sentimentclass {positive, negative, neutral, objective}\n")
        f.write("@data\n")
        for dindex, vec in enumerate(features):
            w = "{ "
            cvec = vec.tocoo()
            values = []
            for i, j, v in itertools.izip(cvec.row, cvec.col, cvec.data):
                values.append((j, v));
            values.sort(key=lambda x: x[0])
            for v in values:
                w += str(v[0]) + " " + str(v[1]) + ", "
            w += str(length) + " " + str(msg_analytics[dindex]["smilies"]) + ", "
            w += str(length + 1) + " " + str(msg_analytics[dindex]["sadfaces"]) + ", "
            w += str(length + 2) + " " + str(msg_analytics[dindex]["exclamations"]) + ", "
            w += str(length + 3) + " " + str(msg_analytics[dindex]["questions"]) + ", "
            w += str(length + 4) + " " + str(msg_analytics[dindex]["posscore"]) + ", "
            w += str(length + 5) + " " + str(msg_analytics[dindex]["negscore"]) + ", "
            w += str(length + 6) + " " + str(msg_analytics[dindex]["objscore"]) + ", "
            w += str(length + 7) + " " + polarity[dindex]
            w += "}\n"
            f.write(w)


to_sparse_arff("results-sparse-with-analytics-pos-neg-obj-words-normalized.arff")

# print len(features[0])
# counter = 0
# features_list = []
# # print (features[0].tolist()[0])
# for index, val in enumerate(features):
#     list = val.tolist()
#     # print ids[index]
#     list.append(polarity[index])
#     # print list
#     features_list.append(list)
#     # if index == 500:
#     #     break
#
#
# names=vectorizer.get_feature_names()
# names.append("category")
#
# arff.dump("result4.arff", features_list, relation="opinion", names=names)


#
# print len(vectorizer.get_feature_names())
# for i in vectorizer.get_feature_names():
#     print i
