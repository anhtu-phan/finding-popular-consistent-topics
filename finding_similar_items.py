import pandas as pd
import pre_process
import os
from numpy import asarray, save, load
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


def get_data(path, column_name):
    df = pd.read_csv(path, delimiter=",")
    tran = []
    try:
        for _, row in df.iterrows():
            if pd.isna(row[column_name]) or pd.isnull(row[column_name]) or row[column_name] == "":
                continue
            texts = row[column_name].split()
            if len(texts) > 0:
                tran.append(texts)
    except Exception as e:
        print(row)
        raise e
    return tran


def check_text_exist(text, list_check):
    if pd.isna(text) or pd.isnull(text) or text == "":
        return False
    for check in list_check:
        if check not in text:
            return False
    return True


def write_result(f, item_set, rules, column_name):
    doc = pd.read_csv("./dataset/covid19_tweets_processed.csv")
    for num, se in item_set.items():
        if num > 1:
            for s in se:
                f.write(str(s) + "\n")
                doc['checking'] = doc[column_name].apply(check_text_exist, args=(s,))
                list_dates = doc.loc[doc['checking'] == True]['date'].array
                f.write("\n".join(list_dates))
                f.write("\n=======================================\n")
    for item in rules:
        f.write(str(item[0]) + " -> " + str(item[1]) + ": " + str(item[2]) + "\n")


def write_result_with_date(f, output, mapping):
    for index, time_range in output.items():
        s_date = ""
        for i in range(0, len(time_range), 2):
            s_date += (datetime.strftime(time_range[i], "%Y-%m-%d")
                       + "->" + datetime.strftime(time_range[i + 1], "%Y-%m-%d") + " ")
        f.write(str(mapping[index]) + ": " + s_date + "\n")


def get_time_range(item_sets):
    tweets = pd.read_csv("./dataset/covid19_tweets_processed_sort_by_date.csv")
    mapping = {}
    for index, s in enumerate(item_sets):
        if len(s) > 1:
            mapping[index] = s
    m_out = {}
    for _, row in tweets.iterrows():
        for index, s in mapping.items():
            if check_text_exist(row[PRE_PROCESS_TYPE], s):
                date_obj = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
                if index not in m_out:
                    m_out[index] = [date_obj, date_obj]
                elif (date_obj - m_out[index][-1]).days <= 1:
                    m_out[index][-1] = date_obj
                else:
                    m_out[index] = m_out[index] + [date_obj, date_obj]
    return m_out, mapping


@profile
def run_fpgrowth(transactions):
    return fpgrowth(transactions, minSupRatio=min_sup, minConf=min_conf)


@profile
def run_pcy(transactions):
    return pcy(transactions, min_sup, 50)


@profile
def run_apriori(transactions):
    return apriori(transactions, min_sup, min_conf)


@profile
def main():
    if not os.path.exists('./dataset/covid19_tweets_processed.csv'):
        print("Running pre process data ...")
        pre_process.process()

    saved_data_path = './dataset/apriori/' + PRE_PROCESS_TYPE + '.npy'
    if not os.path.exists(saved_data_path):
        print("Getting data ...")
        data_path = './dataset/covid19_tweets_processed.csv'
        transactions = asarray(get_data(data_path, PRE_PROCESS_TYPE))
        save(saved_data_path, transactions)
    else:
        print("Load data ...")
        transactions = load(saved_data_path, allow_pickle=True)

    with open("./output/" + alg + "/" + PRE_PROCESS_TYPE + "_" + str(min_sup) + "_" + str(min_conf) + ".txt", 'w') as f:
        f.write("Start time: " + str(datetime.now()) + "\n")
        if alg == 'fpg':
            start_time = time.time()
            freqItemSet, rules = fpgrowth(transactions, minSupRatio=min_sup, minConf=min_conf)
            run_time = time.time() - start_time
        elif alg == "apriori":
            start_time = time.time()
            itemSets, rules = run_apriori(transactions)
            run_time = time.time() - start_time
            freqItemSet = []
            for num, item_set in itemSets.items():
                if num > 1:
                    freqItemSet += item_set
        else:
            start_time = time.time()
            result = run_pcy(transactions)
            run_time = time.time() - start_time
            freqItemSet = []
            for num, val in result.items():
                if num > 1:
                    freqItemSet += val[1]
        f.write("End time: " + str(datetime.now()) + "\n")
        f.write("Run time: " + str(run_time) + "\n")
        m_out, mapping = get_time_range(freqItemSet)
        write_result_with_date(f, m_out, mapping)


if __name__ == '__main__':
    main()
