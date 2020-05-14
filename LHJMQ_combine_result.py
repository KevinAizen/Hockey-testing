#LHJMQ GroinBar analysis
# #Import the package
import numpy as np
from scipy import signal
import pandas as pd
import os
import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# open desired file
path = '.'
extension = 'xlsx'
os.chdir(path)
result = glob.glob('données OKH/*.{}'.format(extension))

# retrieve GroinBar results and plateforme result
os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/results/')
#df_pf = pd.read_pickle('df_pf.pkl')
template = pd.read_excel('GB_Md.xlsx', sheet_name='BestTry')
template = template[template.Name != 'Yanic Duplessis']
template.reset_index(inplace=True, drop=True)

os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar')
# retrieve listname of consent player
with open('listname', 'rb') as f:
    listname = pickle.load(f)

# define a function to convert foot and inch in cm
def impinmetrics(data):
    """Return the value in cm"""
    data = data.astype(str)
    newdata = []
    for i in data:
        inch_in_cm = float(i.split('.')[0]) * CEN_IN_FEET
        feet_in_cm = i.split('.')[1]
        if len(feet_in_cm) > 1:
            l = len(feet_in_cm)
            feet_in_cm = float(feet_in_cm)/(10**l) * CEN_IN_INCH
        else:
            feet_in_cm = float(feet_in_cm) * CEN_IN_INCH
        newdata.append(inch_in_cm + feet_in_cm)
    return newdata

# define a function to convert two columns name (first and last name) into one
def nameinonecell(df, col_first_name, col_last_name):
    """Return the full name of the athlete in only one cell"""
    df['Name'] = df[col_first_name] + ' ' + df[col_last_name]
    df = df.drop([col_first_name, col_last_name], axis=1)
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('Name')))
    df = df.loc[:, cols]
    return df


# Converter for Height and weight
CEN_IN_INCH = 2.54
CEN_IN_FEET = 30.48

# Bring our features variables X
# A) Okanagan Hockey Off-ice testing
os.chdir("/Users/kevinaizen/PycharmProjects/analysis_groinbar/données OKH/")  # Acess to Données Okanagan hockey

X = pd.read_excel('2019 QMJHL OFF-ICE Overall Results Filter.xlsx')
X = X.drop(
    columns=['Overall Rank',
             'Rank',
             'Med Ball Toss',
             'Rank.1',
             'Pro Agility Left',
             'Rank.2',
             'Pro Agility Right',
             'Rank.3',
             'Rank.4',
             'Grip Right',
             'Rank.5',
             'Grip Left',
             'Rank.6',
             'Rank.7',
             'Rank.8',
             'Wingate Decrease',
             'Rank.9']
)
X = nameinonecell(X, 'First Name', 'Last Name')
X['Weight'] = X['Weight']/2.2
X['Height'] = impinmetrics(X.Height)
X['Broad Jump'] = impinmetrics(X['Broad Jump'])
X['Vertical Jump'] = X['Vertical Jump'] * CEN_IN_INCH
X.sort_values('Name', inplace=True)
X = X.reset_index(drop=True)

os.chdir("/Users/kevinaizen/PycharmProjects/analysis_groinbar/")  # Rechange directory to access consent form player

# B) Groin bar testing 'template' matrix
if [X['Name'] == template['Name'] for i in X]:
    print(f'Tous les noms concordent pour les df X et template')
else:
    print(f'Quelques noms ne concordent pas entre les df X et y... à revoir')

# Normalize Gb force
GB_test = list(template.columns.values)
GB_test.remove('Name')
#template[GB_test] = template[GB_test].div(X['Weight'], axis=0)

# Adding feature GB to our data F_score + Imbalance
def f_score(a, b):
    """Takes two columns and compute the F score."""
    return 2 * (a * b) / (a + b)

def imbalance(a: object, b: object) -> object:
    """compute the imbalance score, in percentage."""
    return np.abs((a - b) / a) * 100


X_Gb = pd.DataFrame()
for name, icol in template[GB_test].iteritems():
    b = np.empty(0)  # start an empty array
    if name[-4:] == 'left':
        a = icol
    else:
        b = icol
        X_Gb[name[4:-6]] = f_score(a, b)
        X_Gb[f'imb_{name[4:-6]}'] = imbalance(a, b)

X_Gb.insert(loc=0, column='Name', value=X['Name'])


# Take only player who did sign the consent and non goalie player
os.chdir("/Users/kevinaizen/PycharmProjects/analysis_groinbar/")
consent = pd.read_excel('consent.xlsx', index_col='Name')
consent.sort_index(
    inplace=True
)
consent.reset_index(
    inplace=True
)
consent = consent[consent.Position != 'GOALIE'].dropna(subset=['GB consent', 'OKH consent'])
consent.reset_index(drop=True, inplace=True)
consent.drop(
    columns=['Position',
             'GB consent',
             'OKH consent'],
    inplace=True
)
name = consent['Name'].tolist()

X = pd.concat([consent.set_index('Name'), X.set_index('Name'), X_Gb.set_index('Name')], axis=1, join='inner').reset_index()

# What is our target variable.
os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/données OKH')
y = pd.read_excel('2019 QMJHL ON-ICE RESULTS WITH TEAM NAME.xlsx', header=1)
y = nameinonecell(y, 'First Name', 'Last Name')
y = y[y['Name'].isin(name)].reset_index(drop=True)
y.sort_values('Name', ascending=True, inplace=True)
y.reset_index(
    drop=True,
    inplace=True
)
y.drop(
    columns=['Overall Rank',
             'Team',
             'Position',
             'Rank',
             'Rank.1',
             'Rank.2',
             'Rank.3',
             'Rank.4',
             'Rank.5',
             'Rank.6',
             'Rank.7',
             'Rank.8',
             'Rank.9'],
    inplace=True
)

# What is our variable of interest?
target = ['5F', '30F', '5B', '30B']

# Tells us if certain row do not fitting
if [X['Name'] == y['Name'] for i in X]:
    print(f'Tous les noms concordent pour les df X et y')
    X = X.drop(
        columns='Name'
    )  # Remove name, we do not need them

    y.rename(
        columns={"30M Forward": "30F",
                 "30F-SPLIT 1 (5M)": "5F",
                 "30M Backward": "30B",
                 "30B-SPLIT1 (5M)": "5B"},
        inplace=True
    )
    y = y[target]
else:
    print(f'Quelques noms ne concordent pas entre les df X et y... à revoir')


# take more variable from sprinting tests

y['5_30F'] = y['30F'] - y['5F']
y['5_30B'] = y['30B'] - y['5B']

y['5VF'] = 5/y['5F']
y['30VF'] = 30/y['30F']
y['5_30VF'] = 25/y['5_30F']

y['5VB'] = 5/y['5B']
y['30VB'] = 30/y['30B']
y['5_30VB'] = 25/y['5_30B']

y = y[
    [
        '5F',
        '5_30F',
        '30F',
        '5VF',
        '5_30VF',
        '30VF',
        '5B',
        '5_30B',
        '30B',
        '5VB',
        '30VB',
        '5_30VB'
    ]
]


# Save input variable and target variables
os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/results/')
X.to_excel('X.xlsx')
y.to_excel('y.xlsx')

X.hist(bins=10, figsize=(20,15))
y.hist(bins=10, figsize=(20,15))
X.info()
y.info()
