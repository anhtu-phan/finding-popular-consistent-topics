import pandas as pd
from datetime import datetime


def check_text_exist(text, list_check):
    if pd.isna(text) or pd.isnull(text) or text == "":
        return False
    for check in list_check:
        if check not in text:
            return False
    return True


def write_result_with_date(f, output, mapping):
    for index, time_range in output.items():
        s_date = ""
        for i in range(0, len(time_range), 2):
            s_date += (datetime.strftime(time_range[i], "%Y-%m-%d")
                       + "->" + datetime.strftime(time_range[i + 1], "%Y-%m-%d") + " ")
        f.write(str(mapping[index]) + ": " + s_date + "\n")