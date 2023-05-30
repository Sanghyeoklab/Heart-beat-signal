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
    return dictionary

def auto_correlation(signal):
    signal = [d - np.mean(signal) for d in signal]
    signal_shift = [d - np.mean(signal) for d in signal]
    output = []
    for _ in range(len(signal)):
        output.append(np.sum(np.array(signal) * np.array(signal_shift)))
        signal_shift = [signal_shift[-1]] + signal_shift[:-1]
    return output

def local_maximum(graph, index, gap = 100):
    start = max(index - gap, 0)
    final = min(index + gap, len(graph))
    return np.argmax(graph[start : final]) + start

def is_local_maximum(graph, index, gap = 100):
    local_point = local_maximum(graph, index, gap)
    if local_point == index:
        return True
    else:
        False

def linear_projection(y):
    x = np.linspace(0, len(y) - 1, len(y)).reshape(-1, 1)
    x = np.concatenate((x, np.ones(x.shape)), 1)
    y = np.array(y).reshape(-1, 1).astype(np.float64)
    return np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))