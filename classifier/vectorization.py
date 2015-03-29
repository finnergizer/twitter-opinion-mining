__author__ = 'shaughnfinnerty'

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer
import codecs
import itertools
import pickle
import twitter_msg
import tools
from nltk.corpus import sentiwordnet as swn

class Vectorizer:
    def __init__(self):
        self.twitter_messages = []
        self.feature_matrix_token_counts = None
        self.token_feature_names = []
        self.amount_of_token_features = 0
        self.additional_features_names = []
        self.additional_features = []


    def read_twitter_messages_from_file(self, path_to_file):
        all_twitter_messages = []
        csv_reader = tools.unicode_csv_reader(open(path_to_file), delimiter='\t')
        for row in csv_reader:
            t = twitter_msg.TwitterMessage(row[0], row[1], row[3], row[2])
            all_twitter_messages.append(t)

        self.twitter_messages = all_twitter_messages
        return all_twitter_messages


    def create_feature_matrix_token_counts(self):
        '''
        Create a n by m matrix of n twitter messages with m features representing
        count of preprocessed, stemmed, tokenized words
        :return: n by m feature matrix of n twitter messages and m features (i.e. word tokens)
        '''

        #Create the basic count vectorizer so that we can copy its preprocessor and tokenizer
        basic_vectorizer = CountVectorizer(stop_words='english')
        preprocessor = basic_vectorizer.build_preprocessor();
        tokenizer = basic_vectorizer.build_tokenizer();

        #Create a stemmer for additional processing after preprocessing and tokenizer
        stemmer = EnglishStemmer()

        #Custom analyzer for Count Vectorizer which stems tokens after preprocessing
        def stemming_analyzer(document):
            return map(stemmer.stem, tokenizer(preprocessor(document)))

        vectorizer = CountVectorizer(stop_words='english', min_df=4, analyzer=stemming_analyzer)

        all_twitter_msg_text = [t.msg_text for t in self.twitter_messages]
        self.feature_matrix_token_counts = vectorizer.fit_transform(all_twitter_msg_text)
        self.token_feature_names = vectorizer.get_feature_names()
        self.amount_of_token_features = len(self.token_feature_names)

        return self.feature_matrix_token_counts

    def add_additional_features(self):
        all_additional_features = []
        all_additional_feature_names = ["smilies", "exclamations", "questions",
                                        "sadfaces", "posscore", "negscore",
                                        "objscore"]
        for document_index, twitter_document in enumerate(self.twitter_messages):
            additional_features = {}
            additional_features["smilies"] = twitter_document.msg_text.count(":)") + twitter_document.msg_text.count(":-)") + twitter_document.msg_text.count(":o)") + twitter_document.msg_text.count(":]") + twitter_document.msg_text.count(":3") + twitter_document.msg_text.count(":c)") + 2*twitter_document.msg_text.count(":D") + 2*twitter_document.msg_text.count("C:")
            additional_features["exclamations"] = twitter_document.msg_text.count("!")
            additional_features["questions"] = twitter_document.msg_text.count("?")
            additional_features["sadfaces"] = twitter_document.msg_text.count(":(") + twitter_document.msg_text.count(":-(") + twitter_document.msg_text.count(":c") + twitter_document.msg_text.count(":[") + 2*twitter_document.msg_text.count("D8") + twitter_document.msg_text.count("D;") + 2*twitter_document.msg_text.count("D=") + twitter_document.msg_text.count("DX");
            additional_features["posscore"] = 0
            additional_features["negscore"] = 0
            additional_features["objscore"] = 0
            for word in twitter_document.msg_text.split():
                for synset in swn.senti_synsets(word):
                    additional_features["posscore"] += synset.pos_score()
                    additional_features["negscore"] += synset.neg_score()
                    additional_features["objscore"] += synset.obj_score()
            all_additional_features.append(additional_features)
        self.additional_features = all_additional_features
        self.additional_features_names = all_additional_feature_names

        return all_additional_features

    def to_sparse_arff_file(self, file_path):
        with codecs.open(file_path, "wb", "utf-8") as f:
            f.write("@RELATION opinion\n")
            for name in self.token_feature_names:
                f.write("@ATTRIBUTE " + name + " numeric\n");

            for name in self.additional_features_names:
                f.write("@ATTRIBUTE " + name + " numeric\n");


            f.write("@ATTRIBUTE sentimentclass {positive, negative, neutral, objective}\n")
            f.write("@data\n")
            for doc_index, feature_vector in enumerate(self.feature_matrix_token_counts):
                w = "{ "
                coordinate_vec = feature_vector.tocoo()
                values = []
                for i, j, v in itertools.izip(coordinate_vec.row, coordinate_vec.col, coordinate_vec.data):
                    values.append((j, v));
                values.sort(key=lambda x: x[0])
                for v in values:
                    w += str(v[0]) + " " + str(v[1]) + ", "

                for feature_index, feature_name in enumerate(self.additional_features_names):
                    w += str(self.amount_of_token_features + feature_index) + " " \
                         + str(self.additional_features[doc_index][feature_name]) + ", "

                w += str(self.amount_of_token_features + len(self.additional_features_names)) \
                     + " " + self.twitter_messages[doc_index].polarity
                w += "}\n"
                f.write(w)


filename = 'semeval_twitter_data.txt'

