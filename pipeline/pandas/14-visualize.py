#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(['Weighted_Price'], axis=1)

df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.floor('D')
df.set_index('Date')

df.Close = df.Close.ffill()
df.High = df.High.fillna(df.Close)
df.Low = df.Low.fillna(df.Close)
df.Open = df.Open.fillna(df.Close)
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

df = df.groupby('Date').agg({
    'High': 'max', 'Low': 'min', 'Open': 'mean', 'Close': 'mean',
    'Volume_(BTC)': 'sum', 'Volume_(Currency)': 'sum'
})
df = df.loc['2017-01-01':]
print(df)

#import matplotlib
#matplotlib.use('TKAgg')

df.plot()
plt.show()

