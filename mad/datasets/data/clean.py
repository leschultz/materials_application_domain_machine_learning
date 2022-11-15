from pymatgen.core import Element
import pandas as pd

name = 'Diffusion_Data.csv'
df = pd.read_csv(name)


hosts1 = df['Material compositions 1'].values
hosts2 = df['Material compositions 2'].values
group = []
for i, j in zip(hosts1, hosts2):
    i = Element(i)
    j = Element(j)

    if i.is_transition_metal and j.is_transition_metal:
        group.append('Pure Transition Metals')
    elif (not i.is_transition_metal) and (not j.is_transition_metal):
        group.append('Pure Non-Transition Metals')
    elif (i.is_transition_metal) and (not j.is_transition_metal):
        group.append('Mostly Transition Metals')
    elif (not i.is_transition_metal) and (j.is_transition_metal):
        group.append('Mostly Non-Transition Metals')
    else:
        group.append('NA')

df['group'] = group
df = df[df['group'] != 'Mostly Non-Transition Metals']
df = df[df['group'] != 'Mostly Transition Metals']

df.to_csv('diffusion_pure.csv', index=False)
print(df)
