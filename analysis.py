# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:28:36 2020

@author: natha
"""

import pandas as pd
from datetime import date
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set(color_codes=True)


df = pd.read_csv('complex.csv')

df.replace('', np.nan, inplace=True)
df.dropna(subset=['name'], inplace=True)

df = df[df.party != '.']
df = df[df.party != 'Speaker']
df = df[df.party != 'Crossbench']
df = df[df.text != 'indicated dissent.']
df = df[df.text != 'indicated assent.']
df = df[df.length > 10]

df.loc[df.party == "Labour (Co-op)", "party"] = "Labour"
df.loc[df.party == "Social Democratic & Labour Party", "party"] = "Labour"

df["date"] = pd.to_datetime(df["date"])

clean = df.groupby(['party', 'date']).agg({'score': ['mean'], 'complex': ['mean']})

# aggreate the data by month and party by averaging the score
# clean = df.groupby(['party', pd.Grouper(key='date', freq='D')]).agg({'score': ['mean']})
clean.columns = ['score_mean', 'complex_mean']
clean = clean.reset_index()

# clean['date_ordinal'] = pd.to_datetime(clean['date']).apply(lambda date: date.toordinal())
clean['date_ordinal'] = pd.to_datetime(clean['date']).apply(lambda date: date.toordinal())
#%%
# 2005-5 = 732067
# 2010-5 = 733893
# 2015-5 = 735719
# 2017-6 = 736481
def lr(df):
    d20055 = 732067
    d20105 = 733893
    d20155 = 735719
    d20176 = 736481
    if df["party"] == "Conservative":
        if df["date_ordinal"] in range(d20055, d20105):
            return 14.535
        if df["date_ordinal"] in range(d20105, d20155):
            return 17.54
        if df["date_ordinal"] in range(d20155, d20176):
            return -1.574
        else:
            return -2.607
    if df["party"] == "Democratic Unionist Party":
        if df["date_ordinal"] in range(d20055, d20176):
            return 16.594
        else:
            return 27.252
    if df["party"] == "Green Party":
        if df["date_ordinal"] in range(d20055, d20176):
            return -29.083
        else:
            return -33.971
    if df["party"] == "Labour":
        if df["date_ordinal"] in range(d20055, d20105):
            return -3.09
        if df["date_ordinal"] in range(d20105, d20155):
            return -1.5
        if df["date_ordinal"] in range(d20155, d20176):
            return -18.137
        else:
            return -27.56
    if df["party"] == "Liberal Democrat":
        if df["date_ordinal"] in range(d20055, d20105):
            return 3.212
        if df["date_ordinal"] in range(d20105, d20155):
            return 4.66
        if df["date_ordinal"] in range(d20155, d20176):
            return -16.067
        else:
            return -21.751
    if df["party"] == "Scottish National Party":
        if df["date_ordinal"] in range(d20055, d20176):
            return -24.664
        else:
            return -25.667
    if df["party"] == "Ulster Unionist Party":
        return 3.118
    if df["party"] == "UK Independence Party":
        if df["date_ordinal"] in range(d20055, d20176):
            return -7.789
        else:
            return 2.037
    return -5.929
def gov(df):
    d20105 = 733893
    d20155 = 735719
    if df["party"] == "Conservative":
        if df["date_ordinal"] > d20105:
            return 1
        else:
            return 0
    if df["party"] == "Labour":
        if df["date_ordinal"] < d20105:
            return 1
        else:
            return 0
    if df["party"] == "Liberal Democrat":
        if df["date_ordinal"] in range(d20105, d20155):
            return 1
        else:
            return 0
    return 0

clean["conservative"] = (clean['party'] == "Conservative").astype(int)
clean["labour"] = (clean['party'] == "Labour").astype(int)
clean["after election"] = (clean['date_ordinal'] > 735375).astype(int)
clean["interaction"] = clean["conservative"] * clean["after election"]
clean["left-right"] = clean.apply(lr, axis=1)
clean["government"] = clean.apply(gov, axis=1)

#%%

correlation = np.corrcoef(clean[['conservative', 'after election',\
                                 'left-right', 'government', 'interaction']], rowvar=0)

Y = clean["complex_mean"]
X = clean[['conservative', 'after election', 'left-right', 'government']]
x = clean['conservative']
X = sm.add_constant(X, prepend=False)
model = sm.OLS(Y, X)
comp1 = model.fit()

Y = clean["complex_mean"]
X = clean[['conservative', 'after election', 'left-right', 'government', 'interaction']]
x = clean['conservative']
X = sm.add_constant(X, prepend=False)
model = sm.OLS(Y, X)
comp2 = model.fit()

Y = clean["score_mean"]
X = clean[['conservative', 'after election', 'left-right', 'government']]
x = clean['conservative']
X = sm.add_constant(X, prepend=False)
model = sm.OLS(Y, X)
score1 = model.fit()

Y = clean["score_mean"]
X = clean[['conservative', 'after election', 'left-right', 'government', 'interaction']]
x = clean['conservative']
X = sm.add_constant(X, prepend=False)
model = sm.OLS(Y, X)
score2 = model.fit()

cons = clean[clean.party == "Labour"]

sns.lmplot(x='date_ordinal', y='score_mean', data=cons)

#%%
summary = {}
# writer = pd.ExcelWriter('regression results.xlsx')
for party in set(df["party"]):

    X = df["date_ordinal"][df["party"] == party]
    Y = df["score"][df["party"] == party]
    X_const = sm.add_constant(X, prepend=False)
    
    mod = sm.OLS(Y, X_const)
    res = mod.fit()
    
    results_as_csv = [res.summary2().tables[0], res.summary2().tables[1], res]
    # res.summary2().tables[0].to_excel(writer, sheet_name=party+'0')
    # res.summary2().tables[1].to_excel(writer, sheet_name=party+'1')
    summary[party] = results_as_csv

# writer.save()
#%%
summary = {}
writer = pd.ExcelWriter('complex regression results.xlsx')
for party in set(df["party"]):

    X = df["date_ordinal"][df["party"] == party]
    Y = df["complex"][df["party"] == party]
    X_const = sm.add_constant(X, prepend=False)
    
    mod = sm.OLS(Y, X_const)
    res = mod.fit()
    
    results_as_csv = [res.summary2().tables[0], res.summary2().tables[1], res]
    res.summary2().tables[0].to_excel(writer, sheet_name=party+'0')
    res.summary2().tables[1].to_excel(writer, sheet_name=party+'1')
    summary[party] = results_as_csv
    
writer.save()
#%%
X = df["date_ordinal"]
y = df["score"]
X_const = sm.add_constant(X, prepend=False)

mod = sm.OLS(y, X_const)
res = mod.fit()

fig, ax = plt.subplots()

plt.scatter(X,y,alpha=0.3)
y_predict = res.params[1] + res.params[0]*X
plt.plot(X,y_predict, linewidth=3)
plt.xlabel('Date')
plt.ylabel('Percent Populist Words')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('`%y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
plt.title("Total")
#%%
party = "Scottish National Party"
X = df["date_ordinal"][df["party"] == party]
Y = df["complex"][df["party"] == party]
X_const = sm.add_constant(X, prepend=False)

mod = sm.OLS(Y, X_const)
res = mod.fit()

fig, ax = plt.subplots()

plt.scatter(X,Y,alpha=0.3)
y_predict = res.params[1] + res.params[0]*X
plt.plot(X,y_predict, linewidth=3)
plt.xlabel('Date')
plt.ylabel('Flesch Score')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('`%y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
plt.title(party)
#%%

party = "Liberal Democrat"
res = summary[party][2]
fig, ax = plt.subplots()

X = df["date_ordinal"][df["party"] == party]
Y = df["score"][df["party"] == party]

plt.scatter(X,Y,alpha=0.3)
y_predict = res.params[1] + res.params[0]*X
plt.plot(X,y_predict, linewidth=3)
plt.xlabel('Date')
plt.ylabel('Percent Populist Words')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('`%y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
plt.title(party)


# %%
summary_data = []
for party in set(df["party"]):
    summary_data.append([party] + list(df[df.party == party].describe()["score"])
                        + list(df[df.party == party].describe()["length"]))
summary_data.append(["Total"] + list(df.describe()["score"]) + list(df.describe()["length"]))

summary_df = pd.DataFrame(np.array(summary_data), columns=["party"]+list(df.describe().index.values)+
                                                                         list(df.describe().index.values))

# summary_df.to_csv("summary_data.csv")
#%%
complex_sum = []
for party in set(df["party"]):
    complex_sum.append([party] + list(df[df.party == party].describe()["complex"]))

complex_sum.append(["Total"] + list(df.describe()["complex"]))

complex_df = pd.DataFrame(np.array(complex_sum), columns=["party"]+list(df.describe().index.values))