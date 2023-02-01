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
zname = r'$|y-\hat{y}|/\sigma_{y}$'

paths = find('.', 'assessment.csv')

df = []
for i in paths:
    d = pd.read_csv(i)
    d['group'] = i.split('/')[-3]
    df.append(d)

df = pd.concat(df)
#df = df[df['split'] == 'test']

try:
    if len(sys.argv) > 1:
        df['n'] = df['mat'].apply(lambda x: len(Composition(x).elements))
        df = df[df['n'] == int(sys.argv[1])]
except Exception:
    df = df[(df['group'] == sys.argv[1]) | (df['group'] == 'original')]

df[zname] = abs(df['y']-df['y_pred'])
#df = df[['group', 'dist', 'y_std', zname, 'y']]
df = df.rename({'y_std': yname}, axis='columns')
df[xname] = df['dist'].apply(trans)
df['group'] = df['group'].apply(lambda x: x.replace('_', '\n'))

fig_absres, ax_absres = pl.subplots()
fig_std, ax_std = pl.subplots()
fig, ax = pl.subplots()
for i, j in df[df['split'] == 'test'].groupby(['group']):
    stdy = j[zname].std()
    absres = j[zname].values/stdy
    dist = j[xname].values
    std = j[yname].values/stdy

    ax_absres.scatter(dist, absres, marker='.', label=i)
    ax_std.scatter(dist, std, marker='.', label=i)
    ax.scatter(absres, std, marker='.', label=i)

ax_absres.legend()
ax_std.legend()
ax.legend()

ax_absres.set_xlabel(xname)
ax_std.set_xlabel(xname)
ax.set_xlabel(zname)

ax_absres.set_ylabel(zname)
ax_std.set_ylabel(yname)
ax.set_ylabel(yname)

#pl.show()
print(df)
dfte = df[df['split'] == 'test']

groups = dfte.groupby('group')
median = groups.median()
dist = median[xname].sort_values(ascending=False)
std = median[yname].sort_values(ascending=False)

dist = dist.to_frame().reset_index()['group'].values
std = std.to_frame().reset_index()['group'].values

dfte['group'] = pd.Categorical(dfte['group'], dist)
fig, ax = pl.subplots()
sns.boxplot(data=dfte, x=xname, y='group', ax=ax, palette='Spectral')
fig.savefig('dis_box_test.png')

dfte['group'] = pd.Categorical(dfte['group'], std)
fig, ax = pl.subplots()
sns.boxplot(data=dfte, x=yname, y='group', ax=ax, palette='Spectral')
fig.savefig('std_box_test.png')

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
