import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MatplotlibKeyMap as mk
import os
import SanskritCharts as hc

df_words = pd.read_csv('./ZipfsGraph/sanskrit_words.csv')
print(df_words.head())

def calculate_ngram_frequency(n):
    df_ngram = pd.DataFrame(columns=['ngram', 'Frequency'])
    for i in range(0, len(df_words)):
        word = df_words.loc[i]['word']

        for j in range(0, len(word) - n + 1):
            ngram = word[j:j+n]

            if ngram in df_ngram['ngram'].values:
                df_ngram.loc[df_ngram['ngram'] == ngram, 'Frequency'] += df_words.loc[i]['frequency']
            else:
                df_ngram.loc[len(df_ngram)] = [ngram, df_words.loc[i]['frequency']]

    df_ngram['Probability'] = df_ngram['Frequency'] / df_ngram['Frequency'].sum()
    df_ngram.to_csv('./ngram/ngram_' + str(n) + '.csv', index=False)
    print(str(n) + "-gram frequency calculated")

for i in range(1, 10):
    calculate_ngram_frequency(i)

ngram_dict = {}

for i in range(1, 10):
    ngram_dict[i] = pd.read_csv('./ngram/ngram_' + str(i) + '.csv')

total_chars = ngram_dict[1].shape[0]
print("Total chars: " + str(total_chars))

F0=np.log2(total_chars)

print("F0: " + str(F0))

Fn = []
Fn.append(F0)

F1 = ngram_dict[1]['Probability'].apply(lambda x: x * np.log2(x)).sum() * -1
print("F1: " + str(F1))

Fn.append(F1)

for i in range(2, 10):
    Fi = ngram_dict[i]['Probability'].apply(lambda x: x * np.log2(x)).sum() * -1
    Fi = Fi - Fn[len(Fn) - 1]
    print("F" + str(i) + ": " + str(Fi))
    Fn.append(Fi)

df = pd.DataFrame(columns=['n', 'Fn'])

for i in range(0, len(Fn)):
    df.loc[i] = [i, Fn[i]]

df.to_csv('./ngram/ngram_entropy.csv', index=False)

