from matplotlib import pyplot as pl
from pathlib import Path

import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib
import sys

# Font styles
font = {
        'font.size': 16,
        'lines.markersize': 10,
        'figure.figsize': (16, 9)
        }
matplotlib.rcParams.update(font)


def find(where, match):
    paths = list(map(str, Path(where).rglob(match)))
    return paths


def trans(x):
    return -np.log10((1e-20)-x)


xname = r'$-log_{10}((1e-20)-KDE)$'
yname = r'$\sigma_{c}$'

paths = find('.', 'assessment.csv')

df = []
for i in paths:
    print(i)
    d = pd.read_csv(i)
    d['group'] = i.split('/')[-3]
    df.append(d)

df = pd.concat(df)
df = df[df['split'] == 'test']
print(df)

try:
    if len(sys.argv) > 1:
        df['n'] = df['mat'].apply(lambda x: len(Composition(x).elements))
        df = df[df['n'] == int(sys.argv[1])]
except Exception:
    df = df[(df['group'] == sys.argv[1]) | (df['group'] == 'original')]

df = df[['dist', 'y_std', 'split', 'group', 'in_domain']]

df = df[df['split'] == 'test']
df = df[['group', 'dist', 'y_std']]
df = df.rename({'y_std': yname}, axis='columns')
df[xname] = df['dist'].apply(trans)
df['group'] = df['group'].apply(lambda x: x.replace('_', '\n'))

groups = df.groupby('group')
median = groups.median()
dist = median[xname].sort_values(ascending=False)
std = median[yname].sort_values(ascending=False)

dist = dist.to_frame().reset_index()['group'].values
std = std.to_frame().reset_index()['group'].values

df['group'] = pd.Categorical(df['group'], dist)
fig, ax = pl.subplots()
sns.boxplot(data=df, x=xname, y='group', ax=ax, palette='Spectral')
fig.savefig('dis_box.png')

df['group'] = pd.Categorical(df['group'], std)
fig, ax = pl.subplots()
sns.boxplot(data=df, x=yname, y='group', ax=ax, palette='Spectral')
fig.savefig('std_box.png')

df['group'] = pd.Categorical(df['group'], dist)
fig, ax = pl.subplots()
sns.violinplot(
               data=df,
               x=xname,
               y='group',
               ax=ax,
               palette='Spectral',
               cut=0,
               scale='width',
               inner='quartile'
               )
fig.savefig('dis_violin.png')

df['group'] = pd.Categorical(df['group'], std)
fig, ax = pl.subplots()
sns.violinplot(
               data=df,
               x=yname,
               y='group',
               ax=ax,
               palette='Spectral',
               cut=0,
               scale='width',
               inner='quartile'
               )
fig.savefig('std_violin.png')
