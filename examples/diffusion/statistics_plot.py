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


xname = 'dist'
yname = r'$\sigma_{c}$'
zname = r'$|y-\hat{y}|/\sigma_{y}$'
zname = r'$|y-\hat{y}|$'
zzname = r'$y-\hat{y}$'

paths = find('.', 'assessment.csv')

df = []
for i in paths:
    d = pd.read_csv(i)
    d['group'] = i.split('/')[-3]
    df.append(d)

df = pd.concat(df)
df = df[df['split'] == 'test']

try:
    if len(sys.argv) > 1:
        df['n'] = df['mat'].apply(lambda x: len(Composition(x).elements))
        df = df[df['n'] == int(sys.argv[1])]
except Exception:
    df = df[(df['group'] == sys.argv[1]) | (df['group'] == 'original')]

df[zname] = abs(df['y']-df['y_pred'])
df[zzname] = df['y']-df['y_pred']
df = df.rename({'y_std': yname}, axis='columns')
df['group'] = df['group'].apply(lambda x: x.replace('_', '\n'))

dfte = df[df['split'] == 'test']

groups = dfte.groupby('group')
median = groups.median()
dist = median[xname].sort_values(ascending=False)
std = median[yname].sort_values(ascending=False)
res = median[zname].sort_values(ascending=False)

dist = dist.to_frame().reset_index()['group'].values
std = std.to_frame().reset_index()['group'].values
res = res.to_frame().reset_index()['group'].values

dfte['group'] = pd.Categorical(dfte['group'], dist)
fig, ax = pl.subplots()
sns.boxplot(data=dfte, x=xname, y='group', ax=ax, palette='Spectral')
fig.savefig('dis_box_test.png')

dfte['group'] = pd.Categorical(dfte['group'], std)
fig, ax = pl.subplots()
sns.boxplot(data=dfte, x=yname, y='group', ax=ax, palette='Spectral')
fig.savefig('std_box_test.png')

dfte['group'] = pd.Categorical(dfte['group'], res)
fig, ax = pl.subplots()
sns.boxplot(data=dfte, x=zname, y='group', ax=ax, palette='Spectral')
fig.savefig('res_box_test.png')

dfte['group'] = pd.Categorical(dfte['group'], dist)
fig, ax = pl.subplots()
sns.violinplot(
               data=dfte,
               x=xname,
               y='group',
               ax=ax,
               palette='Spectral',
               cut=0,
               scale='width',
               inner='quartile'
               )
fig.savefig('dis_violin_test.png')

dfte['group'] = pd.Categorical(dfte['group'], std)
fig, ax = pl.subplots()
sns.violinplot(
               data=dfte,
               x=yname,
               y='group',
               ax=ax,
               palette='Spectral',
               cut=0,
               scale='width',
               inner='quartile'
               )
fig.savefig('std_violin_test.png')

dfte['group'] = pd.Categorical(dfte['group'], res)
fig, ax = pl.subplots()
sns.violinplot(
               data=dfte,
               x=zname,
               y='group',
               ax=ax,
               palette='Spectral',
               cut=0,
               scale='width',
               inner='quartile'
               )
fig.savefig('res_violin_test.png')
