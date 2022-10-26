import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv(r"D:\새 폴더 (3)\CTC_102021_short.csv")
df = df.reindex(columns=['truckid', 'readdate', 'x', 'y', 'speed'])
df = df.rename(columns={'truckid': 'id', 'readdate': 't', 'speed': 'v'})
df['t'] = pd.to_datetime(df.t)
df = df.sort_values(by=['id', 't'])

data_per_truck = df.groupby('id').t.count()
b11 = np.linspace(0, 10000, 30)
x11 = (b11[:-1]+b11[1:]) / 2
h11, _ = np.histogram(data_per_truck, b11)
f1 = plt.figure(figsize=(10,6))
a11 = f1.add_subplot(121)
a11.plot(x11, h11, 'o-')
a11.grid()
a11.set_xlabel("Number of Measurements per Truck")
a11.set_ylabel("Counts")
f1.suptitle('13405 Unique Trucks')

b12 = np.logspace(0, 4, 30)
x12 = (b12[:-1]+b12[1:]) / 2
h12, _ = np.histogram(data_per_truck, b12)
a12 = f1.add_subplot(122)
a12.plot(x12, h12, 'o-')
a12.grid()
a12.set_xlabel("Number of Measurements per Truck (log scale)")
a12.set_ylabel("Counts")
a12.set_xscale('log')
plt.tight_layout()


data_per_speed = df.groupby('v').t.count()
f2 = plt.figure()
a2 = f2.add_subplot(111)
a2.plot(data_per_speed.index, data_per_speed, 'o-')
a2.grid()
a2.set_xlabel("Speed")
a2.set_ylabel("Count")
a2.set_title("Data Count per Speed")

f2 = plt.figure()
a2 = f2.add_subplot(111)
a2.plot(data_per_speed[1:].index, data_per_speed[1:], 'o-')
a2.grid()
a2.set_xlabel("Speed")
a2.set_ylabel("Count")
a2.set_title("Data Count per Speed")
plt.tight_layout()


df1 = df.iloc[:-1]
df2 = df.iloc[1:]

ddf = pd.DataFrame()
ddf['id'] = df1.id.values
ddf['id2'] = df2.id.values
ddf['dT'] = (df2.t.values - df1.t.values).astype('timedelta64[s]')
ddf.dT = ddf.dT.dt.total_seconds()
ddf['dx'] = df2.x.values - df1.x.values
ddf['dy'] = df2.y.values - df1.y.values
ddf['v1'] = df1.v.values
ddf['v2'] = df2.v.values

ddf = ddf.loc[ddf.id == ddf.id2]
ddf = ddf.drop(columns='id2')
ddf.index = range(len(ddf.index))

ddf['av'] = (ddf.v1 + ddf.v2) // 2
ddf['dr'] = ddf.av * ddf.dT / 3600

f = plt.figure()
a = f.add_subplot(111)
uid = ddf.id.unique()
for i in range(30):
    q = ddf.loc[ddf.id == uid[i]]
    a.plot(q.dT.cumsum(), q.dr.cumsum())