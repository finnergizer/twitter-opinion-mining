__author__ = 'shaughnfinnerty'
import csv
import pickle
from nltk.corpus import sentiwordnet as swn
import nltk

def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    messages=[]
    csv_reader = csv.reader(utf8_data, delimiter='\t', dialect=dialect,  **kwargs)
    for row in csv_reader:
        # yield [unicode(cell, 'utf-8') for cell in row]
        id = unicode(row[0],'utf-8')
        polarity = unicode(row[2], 'utf-8')
        text = unicode(row[3],'utf-8')
        messages.append({"id": id, "text": text, "polarity": polarity})
    return messages
        # print(row[3])




def count_occurrences(str, search_str):
    return str.count(search_str);

def count_positive_words(str):
    return

def run():
    # print(count_occurrences("Hello!!!! My Name is Shaughn :)!!!", ":)"))
    filename = 'semeval_twitter_data.txt'
    messages = unicode_csv_reader(open(filename))
    print(len(messages))
    for msg in messages:
        msg["smilies"] = msg["text"].count(":)") + msg["text"].count(":-)") + msg["text"].count(":o)") + msg["text"].count(":]") + msg["text"].count(":3") + msg["text"].count(":c)") + 2*msg["text"].count(":D") + 2*msg["text"].count("C:")
        msg["exclamations"] = msg["text"].count("!")
        msg["questions"] = msg["text"].count("?")
        msg["sadfaces"] = msg["text"].count(":(") + msg["text"].count(":-(") + msg["text"].count(":c") + msg["text"].count(":[") + 2*msg["text"].count("D8") + msg["text"].count("D;") + 2*msg["text"].count("D=") + msg["text"].count("DX");
        msg["posscore"] = 0
        msg["negscore"] = 0
        msg["objscore"] = 0
        for word in msg["text"].split():
            for synset in swn.senti_synsets(word):
                msg["posscore"] += synset.pos_score()
                msg["negscore"] += synset.neg_score()
                msg["objscore"] += synset.obj_score()

    pos_scores = [msg["posscore"] for msg in messages]
    neg_scores = [msg["negscore"] for msg in messages]
    obj_scores = [msg["objscore"] for msg in messages]

    pos_max = max(pos_scores)
    neg_max = max(neg_scores)
    obj_max = max(obj_scores)

    for msg in messages:
        msg["posscore"] =  msg["posscore"]/pos_max
        msg["negscore"] =  msg["negscore"]/neg_max
        msg["objscore"] =  msg["objscore"]/obj_max

    info_file = open("msg_analytics_pos-neg-obj-scores-normalized.obj", "wb")
    pickle.dump(messages, info_file);

run()

def printScores(i, messages):
    print messages[i]["text"]
    print messages[i]["polarity"]
    print messages[i]["posscore"]
    print messages[i]["negscore"]
    print messages[i]["objscore"]
    print "\n"

def printCounts(messages):
    print