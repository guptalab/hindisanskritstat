
# %%
import numpy as np
import HindiChars as hc
import pandas as pd
import matplotlib.pyplot as plt
import decimal
import matplotlib as mpl
import matplotlib.font_manager as mfm
import MatplotlibKeyMap as mk
import os
import re
import multiprocessing as mp
import matplotlib
import threading
from IPython.display import display, clear_output


# %%
print(hc.vyanjans)

# %%
print(hc.swars)

# %%
print(hc.matras)

# %%
print(hc.specialCharacters)

# %%
def initialize_dataframes():
    df_words = pd.DataFrame(columns=['word', 'frequency'])
    df_monograms = pd.DataFrame(columns=['monogram', 'frequency'])
    df_digrams = pd.DataFrame(columns=['digram', 'frequency'])
    df_triagrams = pd.DataFrame(columns=['triagram', 'frequency'])
    df_endings = pd.DataFrame(columns=['ending', 'frequency'])
    df_vyanjan = pd.DataFrame(columns=['vyanjan', 'frequency'])
    df_matra = pd.DataFrame(columns=['matra', 'frequency'])
    df_halfLetters = pd.DataFrame(columns=['halfLetter', 'frequency'])
    df_swar = pd.DataFrame(columns=['swar', 'frequency'])
    
    df_vyanjan['vyanjan'] = np.array(hc.vyanjans)
    df_vyanjan['frequency'] = np.zeros(len(hc.vyanjans), dtype=float)
    
    df_matra['matra'] = np.array(hc.matras)
    df_matra['frequency'] = np.zeros(len(hc.matras), dtype=float)
    
    df_swar['swar'] = np.array(hc.swars)
    df_swar['frequency'] = np.zeros(len(hc.swars), dtype=float)
    
    dataframe_dict = {'words': df_words, 
                      'monograms': df_monograms, 
                      'digrams': df_digrams, 
                      'triagrams': df_triagrams, 
                      'endings': df_endings, 
                      'vyanjans': df_vyanjan, 
                      'matras': df_matra, 
                      'halfLetters': df_halfLetters, 
                      'swars': df_swar}
    
    return dataframe_dict    

# %%
def InsertOrIncrease(df, type,word):
    for letter in word:
        if letter not in hc.vyanjans and letter not in hc.matras and letter not in hc.swars and letter not in hc.specialCharacters:
            # print(f'Letter {letter} not found in vyanjans, matras or swars')
            return
    if df[type].isin([word]).any():
        # print(f'Word {word} already exists')
        df.loc[df[type] == word, 'frequency'] += 1
    else:
        # print(f'Word {word} does not exist')
        df.loc[len(df)] = [word, 1]

# %%
def generateCSV_filePath(df, name, filePath):
    # df['probability']=df['frequency'] / df['frequency'].sum()
    pd.options.display.float_format = '{:.5f}'.format
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    df.to_csv(filePath+name, index=False)

# %%
def generateCSVs_filePath(dataframe_dict, thread_number, filePath):
    for key in dataframe_dict:
        generateCSV_filePath(dataframe_dict[key], key+'.csv', filePath+str(thread_number)+'/')



# %%
# Folder Path 
path = "/home/bhaskarc/Documents/NMF2/aajtak_stories"

number_files = len(os.listdir(path))

print(f'Number of files: {number_files}')

# %%
dataframes = {}

for i in range(5):
    dataframes[i] = initialize_dataframes()

# %%
print(dataframes)

# %%
def process_files(args):
    thread_number, startNumber, endNumber = args
    dataframes[thread_number] = initialize_dataframes()
    for i in range(startNumber, endNumber): 
        
        file_path = f"{path}/FileNumber{i+1}.txt"
        # clear_output(wait=True)
        print("Processing file: "+str(i))

        if os.path.isfile(file_path)==False:
            continue
        
        with open(file_path,'r',16777216 ,encoding='utf-8') as f:
            sentence=f.read()
            
            words = sentence.split()
            # print(f'Number of words: {len(words)}')
            
            for word in words:
                for i in range(len(word)):
                    dataframes[thread_number]['vyanjans'].loc[dataframes[thread_number]['vyanjans']['vyanjan'] == word[i], 'frequency'] += 1
                    dataframes[thread_number]['matras'].loc[dataframes[thread_number]['matras']['matra'] == word[i], 'frequency'] += 1
                    
                    if word[i] in hc.swars:
                        if word[i]=='अ' and i+1<len(word) and word[i+1]=='ं':
                            # print(f'अं detected in {word}')
                            dataframes[thread_number]['swars'].loc[dataframes[thread_number]['swars']['swar'] == 'अं', 'frequency'] += 1
                            continue
                        elif word[i]=='अ' and i+1<len(word) and word[i+1]=='ः':
                            # print(f'अः detected in {word}')
                            dataframes[thread_number]['swars'].loc[dataframes[thread_number]['swars']['swar'] == 'अः', 'frequency'] += 1
                        else:
                            dataframes[thread_number]['swars'].loc[dataframes[thread_number]['swars']['swar'] == word[i], 'frequency'] += 1
                    
                    if(word[i]=='्'):
                        # print(f'Half letter {word[i-1]}{word[i]} detected in {word}')
                        InsertOrIncrease(dataframes[thread_number]['halfLetters'], 'halfLetter',word[i-1]+word[i])

                if len(word) ==1:
                    InsertOrIncrease(dataframes[thread_number]['monograms'], 'monogram',word)
                    
                if len(word) ==2:
                    InsertOrIncrease(dataframes[thread_number]['digrams'], 'digram',word)
                    
                if len(word) ==3:
                    InsertOrIncrease(dataframes[thread_number]['triagrams'], 'triagram',word)
                    
                if word[-1]=='।' or word[-1]=='?' or word[-1]=='!' or word[-1]=='.':
                    InsertOrIncrease(dataframes[thread_number]['endings'], 'ending',word[:-1])
                
                InsertOrIncrease(dataframes[thread_number]['words'], 'word',word)
                
    generateCSVs_filePath(dataframes[thread_number], thread_number, '/home/bhaskarc/Documents/NMF2/CSVs/')


if __name__ == "__main__":
    path = "/home/bhaskarc/Documents/NMF2/aajtak_stories"
    number_files = len(os.listdir(path))
    print(f'Number of files: {number_files}')

    dataframes = {}
    for i in range(5):
        dataframes[i] = initialize_dataframes()

    pool = mp.Pool(mp.cpu_count())
    args = [(i, i*200, (i+1)*500) for i in range(20)]
    pool.map(process_files, args)
    pool.close()
    

