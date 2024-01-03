import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MatplotlibKeyMap as mk

filepath="Bhagavad_Gita/"

def calculate_word(filepath):
    df_words=pd.read_csv(filepath+"words.csv")
    entropy=0

    for i in range(len(df_words)):
        p=df_words.iloc[i,2]
        # print(f'Probility of {df_words.iloc[i,0]} is {p}')
        entropy=entropy-p*np.log2(p)

    print(f'Entropy of the {filepath} is {entropy}')

calculate_word("Bhagavad_Gita/")
calculate_word("Ramayana/Ramayana/")
calculate_word("combined/combined/")
calculate_word("../Hindi/results/")

def calculate_word_till(filepath, n):
    df_words=pd.read_csv(filepath+"words.csv")
    entropy=0

    for i in range(len(df_words)):
        p=df_words.iloc[i,2]
        # print(f'Probility of {df_words.iloc[i,0]} is {p}')
        entropy=entropy-p*np.log2(p)

    print(f'Entropy of the {filepath} is {entropy}')

calculate_word_till("combined/combined/", 2547)
calculate_word_till("../Hindi/results/", 5993)