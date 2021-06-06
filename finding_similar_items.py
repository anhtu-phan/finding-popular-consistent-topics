import pandas as pd
import pre_process
import utils
import argparse
import time
from datetime import datetime, timedelta
from memory_profiler import profile
import random
import os

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


@profile
def main():
    start_date = datetime(2020, 7, 24)
    stop_date = datetime(2020, 8, 30)
    with open("./output/" + alg + "/" + PRE_PROCESS_TYPE + "_" + str(min_sup) + "_" + str(min_conf) + ".txt", 'w') as f:
        f.write("Start time: " + str(datetime.now()) + "\n")
        run_time = 0
        avg_run_time = 0
        results = []
        max_iter = 20
        iter_run = 0
        while iter_run < max_iter:
            date_bin = random.randint(2,15)
            end_date = start_date + timedelta(days=date_bin)
            transactions = pre_process.get_input(PRE_PROCESS_TYPE, start_date, end_date)
            f.write(f"-------- From: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} --------\n")
            if alg == 'fpg':
                start_time = time.time()
                freqItemSet, rules = fpgrowth(transactions, minSupRatio=min_sup, minConf=min_conf)
                run_time += (time.time() - start_time)
            elif alg == "apriori":
                start_time = time.time()
                itemSets, rules = apriori(transactions, min_sup, min_conf)
                run_time += (time.time() - start_time)
                freqItemSet = []
                for num, item_set in itemSets.items():
                    if num > 1:
                        freqItemSet += item_set
            elif alg == "pcy":
                start_time = time.time()
                result = pcy(transactions, min_sup, 50)
                run_time += (time.time() - start_time)
                freqItemSet = []
                for num, val in result.items():
                    if num > 1:
                        freqItemSet += val[1]
            else:
                print("NOT SUPPORT THIS ALGORITHM")
                return
            results.append(freqItemSet)
            for item in freqItemSet:
                if len(item) > 1:
                    f.write(str(item)+"\n")
            start_date = end_date
            if end_date > stop_date:
                f.write("Run time: " + str(run_time) + "\n")
                avg_run_time += run_time
                run_time = 0
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
                    f.write(str(item)+",")
                f.write("\n-----------------------------------------------------------\n")
                if len(final_result) > 0:
                    break
                iter_run += 1
                results = []
                start_date = datetime(2020, 7, 24)
        f.write("End time: " + str(datetime.now()) + "\n")
        f.write("Avg Run time: " + str(avg_run_time/max_iter) + "\n")


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
    if not os.path.exists("./output"):
        os.makedirs("output/apriori")
        os.makedirs("output/BTM")
        os.makedirs("output/fpg")
        os.makedirs("output/LDA")
        os.makedirs("output/pcy")
    main()

    # list_min_sup = [0.02, 0.03, 0.04, 0.05]
    # list_min_conf = [0.02, 0.03, 0.04, 0.05]
    # for v in list_min_sup:
    #     min_sup = v
    #     for u in list_min_conf:
    #         min_conf = u
    #         main()
