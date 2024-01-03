import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel('bhagavad-gita.xlsx',sheet_name='Chapter 1')
df

number_of_chapters = 18

def write_chapter(chapter_number):
    df=pd.read_excel('bhagavad-gita.xlsx',sheet_name='Chapter '+str(chapter_number))
    with open('Bhagvad_Gita.txt','a') as f:
        for i in range(len(df)):
            f.write(df['Sanskrit Anuvad'][i])
            f.write('\n')

for i in range(1,number_of_chapters+1):
    write_chapter(i)

