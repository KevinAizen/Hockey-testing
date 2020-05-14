#LHJMQ GroinBar analysis
# #Import the package
import numpy as np
from scipy import signal
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats
NORMALIZE = 'False'

# open desired file
path = '.'
extension = 'xlsx'
os.chdir(path)
result = glob.glob('results/*.{}'.format(extension))

# retrieve GroinBar results and plateforme result
os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/results/')
X = pd.read_excel('X.xlsx', index_col=0)

GroinBar = [
    "AD",
    "imb_AD",
    "AB",
    "imb_AB",
    "IR",
    "imb_IR",
    "ER",
    "imb_ER",
    "Flexion",
    "imb_Flexion",
    "Extension",
    "imb_Extension"
]

if NORMALIZE == "weight":
    normalizer = X["Weight"]
elif NORMALIZE == "IMC":
    normalizer = X["Weight"] / X["Height"] ** 2
elif NORMALIZE == "weight-height":
    normalizer = X["Weight"] * X["Height"]
else:
    normalizer = 1
    print("data not normalized")

X[GroinBar] = X[GroinBar].divide(normalizer, axis=0)
y = pd.read_excel('y.xlsx', index_col=0)

X_variable = X.columns.tolist()
y_variable = y.columns.tolist()

arrays = [y_variable,
          ['r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           'r', 'alpha',
           ]]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['y_variable', 'pearsonr'])

corr_table = pd.DataFrame(columns=index)

for i in X.iloc[:, 1:].columns:
    index = X[i].index[X[i].apply(np.isnan)]
    X_index = X.index.values.tolist()
    list_index = [X_index.index(i) for i in index]
    variable_x = X[i].drop(labels=list_index)
    for c in y.columns:
        variable_y = y[c].drop(labels=list_index)
        r = stats.pearsonr(variable_x, variable_y)
        corr_table.loc[i, (c, 'r')] = r[0]
        corr_table.loc[i, (c, 'alpha')] = r[1]



corr_table = pd.concat([X, y], axis=1)
corr_table = corr_table.corr()

corr_table = corr_table.loc[X_variable, y_variable]

ax = sns.heatmap(
    corr_table,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    linewidths=0.5,
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

