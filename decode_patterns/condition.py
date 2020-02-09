# посмотреть распределение инструментов
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import random
from decode_patterns.data_conversion import parse_csv, Converter, make_numpy_dataset

grps = {}
for i in range(128):
    grps[i] = i // 8


def view_stat(file_name="../patterns_pairs.tsv", limit=30000):
    instruments = []

    with open(file_name) as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
        for row in tsv_reader:
            if limit <= 0:
                break
            limit -= 1
            if int(row[13]) > 127:
                continue
            instruments.append(grps[int(row[13])])
    _ = plt.hist(instruments, bins=list(range(17)))
    plt.title("Histogram of instruments")
    plt.show()


def view_tempo(file_name = "../patterns_pairs.tsv", limit=30000):
    tempos = []
    maxt = 0
    cnt = 0
    with open(file_name) as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
        for row in tsv_reader:
            # if limit <= 0:
            #     break
            # limit -= 1
            maxt = max(maxt, int(row[3]))
            if int(row[3]) > 2000:
                continue
            tempos.append(int(row[3]))

    _ = plt.hist(tempos, bins='auto')
    plt.title("Histogram of tempo")
    plt.show()
    print(maxt, cnt)


if __name__ == '__main__':
    #  view_melodies()
    view_tempo()