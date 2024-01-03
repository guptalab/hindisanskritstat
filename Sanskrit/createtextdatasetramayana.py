import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

kands = ['balakanda', 'ayodhyakand', 'aranyakanda', 'kishkindakanda', 'sundarakanda']

for file in kands:
    df = pd.read_csv(f'Ramayana/{file}.csv')
    with open(f'Ramayana.txt', 'a') as f:
        for i in range(0, len(df)):
            f.write(str(df['content'][i]))
            f.write('\n')