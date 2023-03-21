#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Apr  6 11:59:05 2020

@author: nevena

Cleaning data, data is given in different excel files and in different formats
(some are in one column, other are in tables).
This is 16 years of daily data 

"""

#%%
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# import datetime as dt
#%%
fp1 = "C:Users---"
temp_vode = pd.read_excel(fp1)
temp_vode.head(5)
#%%
"""the goal here is to flatten data which are in tables (like a calendar)"""

temp_vode = temp_vode.drop("DAN", axis=1)
# dropping a column with ordinal numbers, axis=1 means columns
temp_vode_flat = []


brojac = []
for i in range(16):
    brojac.append(i * 38)
# looking at the data, they are made so that the first important row is vodostaj[1],
# then vodostaj[36] and so on, those are the first dates of a year

for k in brojac:
    temp_vode_flat.append(temp_vode.loc[k : k + 30].to_numpy().flatten(order="F"))
    # we append every year to vodostaj_flat, but so that each year is in
    # a column from 1/1 to 12/31
    # turning a 31x12 matrix to a column is done with flatten
    # order F means that the first column comes first and then the second and so on
    # .loc finds rows from k to k+30
    # append appedns everything to the first empty list

#%%
nova_t = []
for k in range(16):
    for j in range(372):
        nova_t.append(temp_vode_flat[k][j])

print(len(nova_t))

#%%
# now the dates to be dropped, making a list called br
# the thing is that in the previous process we made some empty entries
# like there are 31 days in each month
# in it we put indices to be dropped
# first every 31^{st} in the month
k = 61  # this is the 31st February of 2002 that needs to be dropped
# and it is a starting point
br = []
br.append(k)

for m in range(16):
    for j in range(2):
        k = k + 2 * 31
        br.append(k)  # april and june
    k = k + 3 * 31  # september
    br.append(k)  # september
    k = k + 2 * 31  # november
    br.append(k)  # november
    k = k + 3 * 31  # february 2003, 2004...
    br.append(k)  # february 2003, 2004...

del br[80]
#%%
# then February 28th and 29th

k = 30 + 29
n = 30 + 30
br.append(k)
br.append(n)  # this is 2002
br.append(k + 12 * 31)
br.append(n + 12 * 31)  # this is 2003

k = k + 2 * 12 * 31
n = n + 2 * 12 * 31  # 2004, first leap year
br.append(n)

for p in range(3):
    for j in range(3):
        k = k + 12 * 31
        n = n + 12 * 31
        br.append(k)
        br.append(n)
    k = k + 12 * 31
    n = n + 12 * 31  # 2008,...
    br.append(n)  # this will be 2008, 2012, 2016, so p=3
# ending  n is 2015 (and k also)

br.append(n + 12 * 31)
br.append(k + 12 * 31)
# these 2 are 2017

#%%

temp_vod = []
ne_br = []

for m in range(len(nova_t)):
    if m not in br:
        temp_vod.append(nova_t[m])
        ne_br.append(m)
print(len(temp_vod))


#%%
"""same situation for vairable vodostaj"""

fp5 = "..."
vodostaj = pd.read_excel(fp5)

vodostaj.head(5)
#%%

vodostaj = vodostaj.drop("Unnamed: 0", axis=1)
vodostaj_flat = []
brojac = []
for i in range(16):
    brojac.append(1 + i * 35)


for k in brojac:
    vodostaj_flat.append(vodostaj.loc[k : k + 30].to_numpy().flatten(order="F"))

novi_v = []
for k in range(16):
    for j in range(372):
        novi_v.append(vodostaj_flat[k][j])

print(len(novi_v))

vod = []

ne_br = []

for m in range(len(novi_v)):
    if m not in br:
        vod.append(novi_v[m])
        ne_br.append(m)
print(len(vod))

#%%
"""variable padavine is simpler"""
fp7 = ",,,"

padavine = pd.read_excel(fp7)
padavine = padavine["дн.кол. Падавина (L/m2)"]
padavine = list(padavine)

padavine = padavine[::-1]
# already in one column but needs to be reversed

del padavine[5844:]  # erasing 2018, since the other variables don't have that year
print(padavine)

# and also air temperature:
padavine.columns
temp = padavine["ср.темп. Ваздуха T°C)"]
temp = list(temp)
temp = temp[::-1]
len(temp)
del temp[5844:]  # 2018
print(len(temp))

#%%
"""some plotting"""

fp2 = "C:Users/..."

ind = padavine["Unnamed: 0"]
ind = ind.iloc[::-1]
ind = ind.iloc[:5844]
dic = {
    "padavine": padavine,
    "temp_vazduha": temp,
    "temp_vode": temp_vod,
    "vodostaj": vod,
    "time": ind,
}
df3 = pd.DataFrame(dic)
df3 = df3.set_index(df3["time"])
# filling missing values of water level:
df3["vodostaj"] = df3["vodostaj"].interpolate(method="time", limit_direction="forward")
df3["vodostaj"] = df3["vodostaj"].interpolate(method="time", limit_direction="backward")
df3.head()

df3 = df3.drop(
    [
        pd.Timestamp("2004-02-29 00:00:00"),
        pd.Timestamp("2008-02-29 00:00:00"),
        pd.Timestamp("2012-02-29 00:00:00"),
        pd.Timestamp("2016-02-29 00:00:00"),
    ]
)

corr = df3.corr()
sns.heatmap(corr, vmin=-1, vmax=1, cmap="viridis")
plt.savefig(fp2, dpi=300, bbox_inches="tight")

#%%
fp = "C:Users/..."

# to english:
dff = df3.rename(
    columns={
        "padavine": "precipitation",
        "temp_vazduha": "air temperature",
        "temp_vode": "water temperature",
        "vodostaj": "water level",
    }
)

df1 = dff[["precipitation"]]
df4 = dff[["air temperature"]]
df5 = dff[["water temperature"]]
df6 = dff[["water level"]]

df5 = df5.set_index(df3["time"])


ax = df5.plot(colormap="coolwarm")
xtick = pd.date_range(start=df5.index.min(), end=df5.index.max(), freq="Y")
ax.set_xticks(xtick, minor=True)
ax.grid("on", which="minor", axis="x")
ax.grid("off", which="major", axis="x")
fig = ax.get_figure()
fig.savefig(fp, dpi=300)


#%%
"""saving data as tensors for the neural network"""
fp4 = "C:/Users..."

n = len(df3)
p1 = df3["padavine"]
p1 = list(p1)

tv = df3["temp_vazduha"]
tv = list(tv)

v1 = df3["vodostaj"]
v1 = list(v1)

tvode = df3["temp_vode"]
tvode = list(tvode)

p1 = tf.reshape(tf.constant(p1, dtype=tf.float32), (n, 1))
tv = tf.reshape(tf.constant(tv, dtype=tf.float32), (n, 1))
v1 = tf.reshape(tf.constant(v1, dtype=tf.float32), (n, 1))
tvode = tf.reshape(tf.constant(tvode, dtype=tf.float32), (n, 1))

X = tf.concat([p1, tv, v1, tvode], 1)

with open(fp4, "wb") as f:
    pickle.dump([X], f)
