import pandas as pd
import numpy as np
from collections import defaultdict


def get_data(path):
    data = pd.read_excel(path)
    dictionary = defaultdict(lambda : [])
    for T, H in data.values:
        dictionary[T.hour * 60 + T.minute].append(H)
    for key in dictionary.keys():
        dictionary[key] = np.mean(dictionary[key])
    return dictionary, key

def check_None(dictionary, max_time):
    lists = []
    for i in range(max_time):
        if i not in dictionary.keys():
            lists.append(i)
    return lists