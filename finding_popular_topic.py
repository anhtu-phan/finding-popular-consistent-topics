import argparse
import os
import pandas as pd
from gensim import corpora, models
import gensim
from fpgrowth_py import fpgrowth
from pcy import pcy
from apriori_python import apriori
import pre_process
from datetime import datetime, timedelta
import random

ALG_TOPIC = "BTM"
ALG_SIMILAR = "fpg"
PRE_PROCESS_TYPE = 'remove_twitter_account'
NUM_TOPIC = 20


def run_lda(start_date, end_date):
    processed_docs = pre_process.get_input(PRE_PROCESS_TYPE, start_date, end_date)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # tfidf = models.TfidfModel(bow_corpus)
    # corpus_tfidf = tfidf[bow_corpus]

    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=NUM_TOPIC, id2word=dictionary, passes=2, workers=2)

    with open("./output/LDA/{}_topic.txt".format(PRE_PROCESS_TYPE), "w") as f:
        for idx, topic in lda_model.print_topics(-1):
            f.write('Topic: {} \nWords: {}\n'.format(idx, topic))

    topic_set = []
    result = {"text": list(), "topic": list()}
    for i_c, corpus in enumerate(bow_corpus):
        topic = lda_model[corpus]
        topic = [str(index) for index, score in topic if score >= 0.2]
        topic_set.append(topic)
        result["text"].append(processed_docs[i_c])
        result["topic"].append(",".join(topic))

    pd_result = pd.DataFrame(result)
    pd_result.to_csv("./output/LDA/{}_result.csv".format(PRE_PROCESS_TYPE))
    return topic_set


def run_btm(start_date, end_date):
    df = pd.read_csv("./dataset/covid19_tweets_processed.csv")
    with open('./BTM/sample-data/covid19_data.txt', 'w') as f:
        for index, row in df.iterrows():
            text = row[PRE_PROCESS_TYPE]
            if pd.isna(text) or pd.isnull(text) or text == "" or not (
                    start_date <= datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S') <= end_date):
                continue
            f.write(text + '\n')

    os.chdir("./BTM/script")
    os.system("sh runExample.sh " + str(NUM_TOPIC))
    os.chdir("../..")
    f_topic = open('./BTM/output/model/k' + str(NUM_TOPIC) + ".pz_d", "r")
    f_vocab = open('./BTM/output/voca.txt', 'r')
    f_topic_word = open('./BTM/output/model/k' + str(NUM_TOPIC) + '.pw_z', 'r')
    f_topic_result = open('./output/BTM/{}_topic.txt'.format(PRE_PROCESS_TYPE), 'w')
    topic_set = []
    vocab = {}

    for line in f_vocab:
        map_word = line.split()
        if len(map_word) == 2:
            vocab[int(map_word[0])] = map_word[1]

    for i_t, line in enumerate(f_topic_word):
        probs = line.split()
        f_topic_result.write("Topic {}:\n".format(i_t))
        word_in_topic = []
        for i_w, prob in enumerate(probs):
            if float(prob) >= 0.005:
                if i_w in vocab:
                    word_in_topic.append('{:.5f}*"{}"'.format(float(prob), vocab[i_w]))
        f_topic_result.write("Words: {}\n".format(" + ".join(word_in_topic)))
    f_topic_result.close()

    for line in f_topic:
        probs = line.split()
        topic = []
        for i_t, prob in enumerate(probs):
            if float(prob) >= 0.2:
                topic.append(i_t)
        topic_set.append(topic)

    return topic_set


def find_topic_popular(topic_set):
    topic_set_ = []
    for s in topic_set:
        if len(s) > 0:
            topic_set_.append(s)
    if ALG_SIMILAR == 'fpg':
        freqItemSet, rules = fpgrowth(topic_set_, minSupRatio=0.0001, minConf=0.0001)
    elif ALG_SIMILAR == 'apriori':
        itemSets, rules = apriori(topic_set_, 0.0001, 0.0001)
        freqItemSet = []
        for num, item_set in itemSets.items():
            if num > 1:
                freqItemSet += item_set
    elif ALG_SIMILAR == 'pcy':
        result = pcy(topic_set_, 0.0001, 50)
        freqItemSet = []
        for num, val in result.items():
            if num > 1:
                freqItemSet += val[1]
    else:
        print("NOT SUPPORT THIS ALGORITHM")
        return

    return freqItemSet


def main():
    start_date = datetime(2020, 7, 24)
    stop_date = datetime(2020, 8, 30)
    with open("./output/{}/{}_result_with_date.txt".format(ALG_TOPIC, PRE_PROCESS_TYPE), "w") as f:
        results = []
        max_iter = 20
        iter_run = 0
        while iter_run < max_iter:
            date_bin = random.randint(2, 7)
            end_date = start_date + timedelta(days=date_bin)
            if ALG_TOPIC == "LDA":
                topic_set = run_lda(start_date, end_date)
            elif ALG_TOPIC == "BTM":
                topic_set = run_btm(start_date, end_date)
            else:
                print("NOT SUPPORT THIS ALGORITHM")
                return

            f.write(f"-------------- From: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} --------------\n")
            freqItemSet = find_topic_popular(topic_set)
            results.append(freqItemSet)
            for item in freqItemSet:
                if len(item) > 1:
                    f.write(str(item) + "\n")

            start_date = end_date
            if end_date > stop_date:
                final_result = []
                for i in range(len(results)):
                    for item in results[i]:
                        check_other = 0
                        for j in range(len(results)):
                            if j == i:
                                continue
                            for item_i in results[j]:
                                if len(item_i) == len(item):
                                    if len(item - item_i) == 0:
                                        check_other += 1
                        if check_other < len(results) - 1:
                            final_result.append(item)
                f.write(f"Final Results: ")
                for item in final_result:
                    f.write(str(item) + ",")
                f.write("\n-----------------------------------------------------------\n")
                if len(final_result) > 0:
                    break
                iter_run += 1
                results = []
                start_date = datetime(2020, 7, 24)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finding Popular Topics")
    parser.add_argument("--pre_process_type", nargs="?", default="remove_twitter_account")
    parser.add_argument("--alg_similar", nargs="?", default="fpg")
    parser.add_argument("--alg_topic", nargs="?", default="LDA")
    parser.add_argument("--num_topic", nargs="?", type=int, default=20)

    args = parser.parse_args()
    PRE_PROCESS_TYPE = args.pre_process_type
    ALG_SIMILAR = args.alg_similar
    ALG_TOPIC = args.alg_topic
    NUM_TOPIC = args.num_topic
    main()
