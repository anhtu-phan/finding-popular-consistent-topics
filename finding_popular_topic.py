import os
import time
import pre_process
import pandas as pd
from GSDMM.GSDMM import GSDMM


ALG = "GSDMM"
PRE_PROCESS_TYPE = 'remove_twitter_account'


def build_gsdmm_input(token_path, vocab_path):
    dataset = pd.read_csv("./dataset/covid19_tweets_processed_sort_by_date.csv")
    if not os.path.exists(token_path):
        vocab_id = 0
        vocab = {}
        f_token = open(token_path, "w")
        f_vocab = open(vocab_path, "w")
        for index, row in dataset.iterrows():
            tweet = row[PRE_PROCESS_TYPE]
            if not (pd.isna(tweet) or pd.isnull(tweet) or tweet == ""):
                token_ids = []
                for word in tweet.split():
                    if word in vocab:
                        token_ids.append(str(vocab[word]))
                    else:
                        token_ids.append(str(vocab_id))
                        vocab[word] = vocab_id
                        f_vocab.write('["' + word + '", ' + str(vocab_id) + "]\n")
                        vocab_id += 1
                f_token.write('"docid": {}, "tokenids": [{}]\n'.format(index, ",".join(token_ids)))
        f_token.close()
        f_vocab.close()


def run_gsdmm():
    token_path = "./dataset/GSDMM/"+PRE_PROCESS_TYPE+"_tokens.json"
    vocab_path = "./dataset/GSDMM/"+PRE_PROCESS_TYPE+"_vocab.json"
    build_gsdmm_input(token_path, vocab_path)
    model = GSDMM(token_path, vocab_path, n_iterations=100)

    model.inference()

    model.write_topic_assignments("./output/"+ALG+"/topic-assignments.json")
    model.write_topic_top_words("./output/"+ALG+"/topic-top-20-words.txt")


def run():
    if not os.path.exists('./dataset/covid19_tweets_processed.csv'):
        print("Running pre process data ...")
        pre_process.process()

    if ALG == "GSDMM":
        start_time = time.time()
        run_gsdmm()
        elapsed_time = time.time() - start_time
    # elif ALG = "BTM":

    print("Run time: ", elapsed_time)