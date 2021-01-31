import pandas as pd
import pre_process
import utils
import argparse
import time
from datetime import datetime
from memory_profiler import profile

from pcy import pcy
from apriori_python import apriori
from fpgrowth_py import fpgrowth

PRE_PROCESS_TYPE = "remove_twitter_account"
alg = "fpg"
min_sup = 0.01
min_conf = 0.01


def get_stop_words():
    stop_words = []
    with open("dataset/stopwords.txt", 'r') as f:
        for line in f:
            stop_words.append(line.rstrip())
    return stop_words


def write_result(f, item_set, rules, column_name):
    doc = pd.read_csv("./dataset/covid19_tweets_processed.csv")
    for num, se in item_set.items():
        if num > 1:
            for s in se:
                f.write(str(s) + "\n")
                doc['checking'] = doc[column_name].apply(utils.check_text_exist, args=(s,))
                list_dates = doc.loc[doc['checking'] == True]['date'].array
                f.write("\n".join(list_dates))
                f.write("\n=======================================\n")
    for item in rules:
        f.write(str(item[0]) + " -> " + str(item[1]) + ": " + str(item[2]) + "\n")


def get_time_range(item_sets):
    tweets = pd.read_csv("./dataset/covid19_tweets_processed_sort_by_date.csv")
    mapping = {}
    for index, s in enumerate(item_sets):
        if len(s) > 1:
            mapping[index] = s
    m_out = {}
    for _, row in tweets.iterrows():
        for index, s in mapping.items():
            if utils.check_text_exist(row[PRE_PROCESS_TYPE], s):
                date_obj = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
                if index not in m_out:
                    m_out[index] = [date_obj, date_obj]
                elif (date_obj - m_out[index][-1]).days <= 1:
                    m_out[index][-1] = date_obj
                else:
                    m_out[index] = m_out[index] + [date_obj, date_obj]
    return m_out, mapping


@profile
def main():
    transactions, _ = pre_process.get_input(PRE_PROCESS_TYPE)
    with open("./output/" + alg + "/" + PRE_PROCESS_TYPE + "_" + str(min_sup) + "_" + str(min_conf) + ".txt", 'w') as f:
        f.write("Start time: " + str(datetime.now()) + "\n")
        if alg == 'fpg':
            start_time = time.time()
            freqItemSet, rules = fpgrowth(transactions, minSupRatio=min_sup, minConf=min_conf)
            run_time = time.time() - start_time
        elif alg == "apriori":
            start_time = time.time()
            itemSets, rules = apriori(transactions, min_sup, min_conf)
            run_time = time.time() - start_time
            freqItemSet = []
            for num, item_set in itemSets.items():
                if num > 1:
                    freqItemSet += item_set
        elif alg == "pcy":
            start_time = time.time()
            result = pcy(transactions, min_sup, 50)
            run_time = time.time() - start_time
            freqItemSet = []
            for num, val in result.items():
                if num > 1:
                    freqItemSet += val[1]
        else:
            print("NOT SUPPORT THIS ALGORITHM")
            return

        f.write("End time: " + str(datetime.now()) + "\n")
        f.write("Run time: " + str(run_time) + "\n")
        m_out, mapping = get_time_range(freqItemSet)
        utils.write_result_with_date(f, m_out, mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finding Similar Items")
    parser.add_argument("--pre_process_type", nargs="?", default="remove_twitter_account")
    parser.add_argument("--alg", nargs="?", default="fpg")
    parser.add_argument("--min_sup", nargs="?", type=float, default=0.01)
    parser.add_argument("--min_conf", nargs="?", type=float, default=0.01)

    args = parser.parse_args()
    PRE_PROCESS_TYPE = args.pre_process_type
    alg = args.alg
    min_sup = args.min_sup
    min_conf = args.min_conf

    main()
