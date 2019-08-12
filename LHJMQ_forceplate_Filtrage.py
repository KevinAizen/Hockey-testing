#LHJMQ GroinBar analysis
#Import the package
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import numpy as np
import pickle

with open('listname', 'rb') as f:
    listname = pickle.load(f)

# Path creation and data collection
DATA_PATH = Path('./force_plate/QJMHL.csv')

# data collection
df_pf = pd.read_csv(DATA_PATH, header=6)
df_pf = df_pf.dropna(axis=1, how='any')
df_pf['Athlete'] = df_pf['Athlete'].str.split(', ').str[::-1].str.join(' ')
dftodel= df_pf.filter(like='Landing', axis=1).columns.values
df_pf = df_pf.drop(dftodel, axis=1)
df_pf = df_pf.drop(['Test Type', 'Test Date', 'Body Weight [kg]', 'Trial'], axis=1)
#df = df.set_index('Athlete')

# seulement garder les row où le nom est présent dans listname
df_pf = df_pf.sort_values('Athlete')
df_pf = df_pf[df_pf['Athlete'].isin(listname)].reset_index(drop=True)

#Save the dataframe for variables x
df_pf.to_pickle('df_pf.pkl')

print(df_pf.shape)


# Find the outliers for each columns with Z score
df_pf_o = df_pf
sns.boxplot(x=df_pf['Braking Phase Duration:Concentric Duration'])
z = np.abs(stats.zscore(df_pf.loc[:, df_pf.columns != 'Athlete']))

# Find the outliers for each columns with interquartile
df_pf_o1 = df_pf_o
Q1 = df_pf_o1.quantile(0.25)
Q3 = df_pf_o1.quantile(0.75)
IQR = Q3 - Q1

threshold = 3
cols_to_include = df_pf_o.columns[df_pf_o.columns != 'Athlete'].tolist()

z = pd.DataFrame(z, columns=cols_to_include)

# Manage the outliers
for i in cols_to_include:
    to_remove = z[z[i] >= threshold].index.tolist()
    df_pf_o.loc[to_remove, i] = df_pf_o[i].mean()
    df_pf_o[i].replace(0, df_pf_o[i].mean(), inplace=True)

sns.boxplot(x=df_pf['Braking Phase Duration:Concentric Duration'])

# Get the better, worst or the mean value from variable for each player

timecols_minimumgood = [col for col in df_pf_o.columns
                        if '[ms]' in col
                        or '[s]' in col
                        ]
maximum_goodcol = [col for col in df_pf_o.columns
                   if '[Ns]' in col
                   or '[N]' in col
                   or '[W/kg]' in col
                   or '[W]' in col
                   or '[N/kg]' in col
                   or '[m/s]' in col
                   or '[N/s]' in col
                   or '[N/s/kg]' in col
                   or '[W/s/kg]' in col
                   or '[W/s]' in col
                   or '[in]' in col
                   or '[m/s]' in col
                   or '[J]' in col
                   or '[m/s]' in col
                   ]
dontknow_col = [col for col in df_pf_o.columns
                if '[cm]' in col
                or '[%]' in col
                ]

result_force_pfmean = pd.DataFrame(columns=cols_to_include)
result_force_pfmax = pd.DataFrame(columns=cols_to_include)
result_force_pfmin = pd.DataFrame(columns=cols_to_include)

for name in listname:
    player_value = df_pf_o.loc[df_pf_o['Athlete'] == name]
    for col in cols_to_include:
        result_force_pfmean.loc[name, col] = player_value[col].mean()
        result_force_pfmax.loc[name, col] = player_value[col].max()
        result_force_pfmin.loc[name, col] = player_value[col].min()

