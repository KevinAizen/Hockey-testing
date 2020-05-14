# Distribution graph and correlation

#Import the package

import numpy as np
from scipy import stats
from scipy.stats import norm
import pandas as pd
import os
import glob
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
from scipy.stats import norm
from scipy.stats import pearsonr

from sklearn import preprocessing
import plotly.graph_objects as go

threshold = 4
CEN_IN_INCH = 2.54
CEN_IN_FEET = 30.48

def impinmetrics(data):
    """Return the value in cm"""
    data = data.astype(str)
    newdata = []
    for i in data:
        inch_in_cm = float(i.split('.')[0]) * CEN_IN_FEET
        feet_in_cm = float(i.split('.')[1]) * CEN_IN_INCH
        newdata.append(inch_in_cm + feet_in_cm)
    return newdata

sns.set(style='whitegrid', context='paper')

# Take only player who who sign the consent form
os.chdir("/Users/kevinaizen/PycharmProjects/analysis_groinbar/")
consent = pd.read_excel('consent.xlsx', index_col='Name')
consent.sort_index(
    inplace=True
)
consent.reset_index(
    inplace=True
)

# retrieve GroinBar results and plateforme result
os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/')
#template = pd.read_pickle('template.pkl')
template = pd.read_excel('templatetest1.xlsx')
columns_template = list(template)
template.reset_index(
    inplace=True
)
template = consent.merge(template, how='inner', on=['Name'])
template.dropna(
    subset=['GB consent'],
    inplace=True
)
template.dropna(
    how='all',
    subset=columns_template,
    inplace=True
)
template.reset_index(
    drop=True,
    inplace=True
)



os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/données OKH')

def nameinonecell(df, col_first_name, col_last_name):
    """Return the full name of the athlete in only one cell"""
    df['Name'] = df[col_first_name] + ' ' + df[col_last_name]
    df = df.drop([col_first_name, col_last_name], axis=1)
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('Name')))
    df = df.loc[:, cols]
    return df

off_ice = pd.read_excel('2019 QMJHL OFF-ICE Overall Results Filter.xlsx')

off_ice = nameinonecell(off_ice, 'First Name', 'Last Name')
off_ice['Height'] = impinmetrics(off_ice['Height'])
off_ice['Weight'] = off_ice['Weight']/2.2
off_ice.sort_values('Name', ascending=True, inplace=True)
off_ice.reset_index(
    drop=True,
    inplace=True
)
goalie_list = []
for index in range(len(off_ice['Position'])):
    if off_ice.iloc[index, 1] == 'GOALIE':
        goalie_list.append(off_ice.iloc[index, 0])

#drop goalies from the template
for name in goalie_list:
    template = template[template['Name'] != name]

template.reset_index(
    inplace=True,
    drop=True
)

template.drop(
    columns=['OKH consent', 'GB consent'],
    inplace=True
)

player_weight = off_ice[['Name', 'Weight']]
player_weight.loc[:, 'Weight'] = player_weight['Weight']/2.2

consent.rename({'Name': 'Full name'}, axis=1, inplace=True)
off_ice = pd.concat([consent, off_ice], axis=1, sort=True)

off_ice.drop(
    off_ice[off_ice['Position'] == 'GOALIE'].index,
    inplace=True
)

off_ice.dropna(
    subset=['OKH consent'],
    inplace=True
)

off_ice.reset_index(inplace=True,
                    drop=True)

off_ice.drop(
    columns=['Full name',
             'GB consent',
             'OKH consent',
             'Overall Rank',
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

off_ice_noname = off_ice.drop(
    columns='Name'
)

columns_off_ice = list(off_ice_noname)

on_ice = pd.read_excel('2019 QMJHL ON-ICE RESULTS WITH TEAM NAME.xlsx', header=1)

on_ice = nameinonecell(on_ice, 'First Name', 'Last Name')
on_ice.sort_values('Name', ascending=True, inplace=True)
on_ice.reset_index(
    drop=True,
    inplace=True
)
consent.rename({'Full name': 'Name'}, axis=1, inplace=True)

on_ice = consent.merge(on_ice, how='inner', on=['Name'])

on_ice.dropna(
    subset=['OKH consent'],
    inplace=True
)

on_ice.reset_index(inplace=True,
                    drop=True)

on_ice = on_ice.drop(
    columns=['GB consent',
             'OKH consent',
             'Overall Rank',
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
             'Rank.9']
)

os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar')

on_ice_noname = on_ice.drop(
    columns='Name')

columns_on_ice = list(on_ice_noname)

name = off_ice.iloc[:, 0]



#Normalize off_ice data
standardised_off_ice = stats.zscore(off_ice_noname)
standardised_on_ice = stats.zscore(on_ice_noname)

# Generate distribution for OKH testing.
# 1) All Off ice testing
# 1A) Normalized histrogram
Off_ice_dist, axs = plt.subplots(nrows=6,
                                 ncols=2,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(10, 7))
Off_ice_dist.subplots_adjust(hspace=0.4, wspace=0.4)
plt.yticks(np.arange(0, 1, step=0.20))
plt.xlim((-4, 4))
Off_ice_dist.text(0.08, 0.5, 'Density of probalility', va='center', rotation='vertical')
Off_ice_dist.text(0.5, 0.06, 'Zscore', va='center', rotation='horizontal')

Off_ice_dist.suptitle('Distribution of Off-ice testing by Okanagan Hockey (Zscore)', fontsize=16)

axy = 0
axx = 0

for test in columns_off_ice:
    if columns_off_ice.index(test) > 5:
        axy = 1
        axx = 6
    sns.distplot(
        standardised_off_ice[:, columns_off_ice.index(test)],
        rug=True,
        fit=norm,
        kde_kws={"label": "KDE"},
        fit_kws={"color": "r", "lw": 1, "alpha": 0.6, "label": "Norm"},
        norm_hist=False,
        ax=axs[columns_off_ice.index(test) - axx, axy],
    )
    axs[columns_off_ice.index(test) - axx, axy].set_title(test)
    axs[columns_off_ice.index(test) - axx, axy].legend(loc='best')


plt.savefig('./fig/standadised_dist_off_ice.png', dpi=500)

# 1B) Box plot of off ice data standardised

# I need to melt the data to make the box plot, but with standardised data
plt.figure(figsize=(12, 7))
off_ice_zscore = pd.DataFrame(columns=columns_off_ice)
outliers_off_ice = pd.DataFrame()
for test in off_ice_noname:
    loc = columns_off_ice.index(test)
    off_ice_zscore[test] = standardised_off_ice[:, loc]
    for i in range(len(off_ice_zscore[test])):
        if np.abs(off_ice_zscore.iloc[i, loc]) >= threshold:
            indice = i
            app_df = {'Player name': [name[i]],
                      'index': [i],
                      'test': [test],
                      'Value zscore': [off_ice_zscore.iloc[i, loc]],
                      'Value abs': [off_ice_noname.iloc[i, loc]]
                      }
            outliers_off_ice = outliers_off_ice.append(pd.DataFrame(data=app_df))


for i in range(len(outliers_off_ice)):
    test_i = columns_off_ice.index(outliers_off_ice.iloc[i, 2])
    row_i = outliers_off_ice.iloc[i, 1]
    off_ice_zscore.iloc[row_i, test_i] = off_ice_zscore.replace(
        off_ice_zscore.iloc[row_i, test_i],
        np.nan,
        inplace=True
    )
    off_ice_noname.replace(
        off_ice_noname.iloc[row_i, test_i],
        np.nan,
        inplace=True
    )
melt_off_ice = off_ice_zscore.melt()
sns.boxplot(x="value",
            y="variable",
            data=melt_off_ice,
            color='grey'
            )
plt.title("Normalized boxplot of off ice testing made by Okanagan Hockey", loc="center", fontsize=16)
plt.xlabel("Zscore")
plt.ylabel("Tests")
plt.xlim((-4, 4))

# Save the Boxplot
plt.savefig('./fig/BP_off_ice.png', dpi=500)

#1C) Descriptive statistic for off-ice et one ice test
off_ice_descriptive = off_ice_noname.describe()
cv = off_ice_descriptive.iloc[2, :]/off_ice_descriptive.iloc[1, :] * 100
cv.name = 'cv %'
off_ice_descriptive = off_ice_descriptive.append(cv, ignore_index=False)
off_ice_descriptive = off_ice_descriptive.round(2)
off_ice_descriptive.drop(off_ice_descriptive.index[4:7], inplace=True)
off_ice_descriptive.columns = [
    'Height (foot.inch)',
    'Weight (kg)',
    'Broad Jump (foot)',
    'Med Ball Toss (foot)',
    'Pro Agility Left (sec)',
    'Pro Agility Right (sec)',
    'Vertical Jump (inch)',
    'Grip Right ()',
    'Grip Left ()',
    '20 M Sprint (sec)',
    'Wingate (W/kg)',
    'Wingate Decrease (%)'
]
# Make a table figure
from pandas.plotting import table
fig, ax = plt.subplots(figsize=(14, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, off_ice_descriptive, loc='center')  # where df is your data frame
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(7) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1) # change size table

plt.savefig('./fig/summary_off_ice.png', dpi=500)





















# 2) All On ice testing
# Preprocess data
# Remove one ice testing with puck
on_ice_todel = list(on_ice_noname.filter(like='Puck', axis=1))
on_ice_noname_nopuck = on_ice_noname.drop(
    columns=on_ice_todel
)
columns_on_ice_nopuck = list(on_ice_noname_nopuck)

index_to_del = []
for i in on_ice_todel:
    index_to_del.append(on_ice_noname.columns.get_loc(i))

# Standardised the one_ice data
standardised_on_ice_nopuck = np.delete(
    standardised_on_ice,
    index_to_del,
    axis=1,
)

# 2A) Distribution of on-ice testing - all test
on_ice_dist, axs = plt.subplots(nrows=3,
                                 ncols=3,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(10, 7))

on_ice_dist.subplots_adjust(hspace=0.4, wspace=0.4)
on_ice_dist.suptitle('Distribution of On-ice testing by Okanagan Hockey', fontsize=16)
on_ice_dist.text(0.08, 0.5, 'Density of probalility', va='center', rotation='vertical')
on_ice_dist.text(0.5, 0.06, 'Zscore', va='center', rotation='horizontal')
plt.xlim((-4, 4))
axy = 0
axx = 0

for test in columns_on_ice_nopuck:
    index = columns_on_ice_nopuck.index(test)
    if (index > 2 and index < 6):
        axy = 1
        axx = 3
    elif index >= 6:
        axy = 2
        axx = 6
    sns.distplot(
        standardised_on_ice[:, index],
        rug=True,
        fit=norm,
        kde_kws={"label": "KDE"},
        fit_kws={"color": "r", "lw": 1, "alpha": 0.6, "label": "Norm"},
        norm_hist=False,
        ax=axs[index - axx, axy]
    )
    axs[index - axx, axy].set_title(test)
    axs[columns_on_ice_nopuck.index(test) - axx, axy].legend(loc='best')

# Save the Distplot
plt.savefig('./fig/standadised_dist_on_ice.png', dpi=500)


# 2B) Boxplot all test
plt.figure(figsize=(12, 7))
on_ice_zscore = pd.DataFrame(columns=columns_on_ice_nopuck)
outliers_on_ice = pd.DataFrame()
for test in on_ice_noname_nopuck:
    loc = columns_on_ice_nopuck.index(test)
    on_ice_zscore[test] = standardised_on_ice_nopuck[:, loc]
    for i in range(len(on_ice_zscore[test])):
        if np.abs(on_ice_zscore.iloc[i, loc]) >= threshold:
            indice = i
            app_df = {'Player name': [name[i]],
                      'index': [i],
                      'test': [test],
                      'Value zscore': [on_ice_zscore.iloc[i, loc]],
                      'Value abs': [on_ice_noname_nopuck.iloc[i, loc]]
                      }
            outliers_on_ice = outliers_on_ice.append(pd.DataFrame(data=app_df), ignore_index=True)

for i in range(len(outliers_on_ice)):
    test_i = columns_on_ice_nopuck.index(outliers_on_ice.iloc[i, 2])
    row_i = outliers_on_ice.iloc[i, 1]
    on_ice_zscore.iloc[row_i, test_i] = np.nan
    on_ice_noname_nopuck.iloc[row_i, test_i] = np.nan

melt_on_ice = on_ice_zscore.melt()
sns.boxplot(x="value",
            y="variable",
            data=melt_on_ice,
            color='grey'
            )
plt.title("Normalized boxplot of on-ice testing made by Okanagan Hockey", loc="center", fontsize=16)
plt.xlabel("Zscore")
plt.ylabel("Tests")
plt.xlim((-4, 4))

# Save the Boxplot
plt.savefig('./fig/BP_on_ice.png', dpi=500)

 #2) Descriptive statistic for off-ice et one ice test
on_ice_descriptive = on_ice_noname_nopuck.describe()
on_ice_descriptive_6 = on_ice_noname_nopuck.iloc[:, :5].describe()
cv = on_ice_descriptive.iloc[2, :]/on_ice_descriptive.iloc[1, :] * 100
cv.name = 'cv %'
on_ice_descriptive = on_ice_descriptive.append(cv, ignore_index=False)
on_ice_descriptive = on_ice_descriptive.round(2)
on_ice_descriptive.drop(on_ice_descriptive.index[4:7], inplace=True)
on_ice_descriptive.columns = [
    '30M-F (sec)',
    '5M-F (sec)',
    '25M-F (sec)',
    '30M-B (sec)',
    '5M-B (sec)',
    '25M-B (sec)',
    'Reaction (sec)',
    'Weave Agility (sec)',
    'Transition Agility (sec)',
]

# Make a table figure
fig, ax = plt.subplots(figsize=(14, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, on_ice_descriptive, loc='center')  # where df is your data frame
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(8) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1) # change size table

plt.savefig('./fig/summary_on_ice.png', dpi=500)













# 3) GroinBar testing
name_GB = template.iloc[:, 0]

# 3)A) Create figure with the distribution of data to compare left and right side.
laterality_distribution_GB = plt.figure(figsize=(12, 7))
plt.title("Left VS Right distribution (Newton)", fontsize=16)

v = sns.violinplot(
    x="test",
    y="value",
    hue="Laterality",
    split=True,
    inner="box",
    scale="count",
    palette="muted",
    data=template.iloc[:, 1:].melt().assign(
        test=lambda x: x["variable"].str.split(' ').str[1],  # prend les string avant le split "/"
        Laterality=lambda x: x["variable"].str.split(' ').str[2]  # take the string after the split string
    ),
)
v.set(xlabel="", ylabel="Force (Newton)")
sns.despine(left=True)

# save the figure
plt.savefig('./fig/violin_GB.png', dpi=500)


# 3)B) Make a result force, combining both left right force with final score, make distribution histogram and box plot
def f_score(a, b):
    """Takes two columns and compute the F score."""
    return 2 * (a * b) / (a + b)

def imbalance(a: object, b: object) -> object:
    """compute the imbalance score, in percentage."""
    return ((a - b) / a) * 100

template_Fscore = pd.DataFrame()
template_imb = pd.DataFrame()

for test, icol in template.iloc[:, 2:].iteritems():
    b = np.empty(0)  # start an empty array
    if test[-4:] == 'left':
        a = icol
    else:
        b = icol
        template_Fscore[test[4:-6]] = f_score(a, b)
        template_imb[f'imb_{test[4:-6]}'] = imbalance(a, b)

# 3)C) Distribution Fscore not normalize
columns_Fscore = list(template_Fscore)
Fscore_distribution, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
Fscore_distribution.text(0.06, 0.5, '% of the data', va='center', rotation='vertical')
Fscore_distribution.text(0.5, 0.03, 'Force (Newton)', va='center', rotation='horizontal')
Fscore_distribution.suptitle('Distribution of GroinBar testing', fontsize=16)
for test in template_Fscore:
    index = columns_Fscore.index(test)
    template_Fscore[test].dropna(axis=0, inplace=True)
    if index > 3:
        axx = 4
        axy = 2
    elif index > 1 and index < 4:
        axx = 3
        axy = 1
    elif index < 2:
        axx = 1
        axy = 0
    sns.distplot(
        template_Fscore[test],
        rug=True,
        fit=norm,
        kde_kws={"label": "KDE"},
        fit_kws={"color": "r", "lw": 1, "alpha": 0.6, "label": "Norm"},
        norm_hist=False,
        ax=axs[index - axx, axy]
    )
    axs[index - axx, axy].legend(loc='best')
    axs[index - axx, axy].set_title(test)

# Save the distribution not normalize
plt.savefig('./fig/distribution_Fscore.png', dpi=500)

# 3)D) distribution Fscore  normalize
Norm_Fscore_distribution, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
Norm_Fscore_distribution.text(0.06, 0.5, 'Density of probability', va='center', rotation='vertical')
Norm_Fscore_distribution.text(0.5, 0.03, 'Zscore', va='center', rotation='horizontal')
Norm_Fscore_distribution.suptitle('Normalized distribution of GroinBar testing', fontsize=16)
plt.xlim((-4, 4))
for test in template_Fscore:
    index = columns_Fscore.index(test)
    template_Fscore[test].dropna(axis=0, inplace=True)
    standardised = stats.zscore(template_Fscore[test])
    if index > 3:
        axx = 4
        axy = 2
    elif index > 1 and index < 4:
        axx = 3
        axy = 1
    elif index < 2:
        axx = 1
        axy = 0
    sns.distplot(
        standardised,
        rug=True,
        fit=norm,
        kde_kws={"label": "KDE"},
        fit_kws={"color": "r", "lw": 1, "alpha": 0.6, "label": "Norm"},
        norm_hist=False,
        ax=axs[index - axx, axy]
    )
    axs[index - axx, axy].legend(loc='best')
    axs[index - axx, axy].set_title(test)

# Save the figure of normalized distribution of the Fscore GroinBar test
plt.savefig('./fig/distribution_GB_normalized.png', dpi=500)

 #3) Descriptive statistic for Zscore
Fscore_descriptive = template_Fscore.describe()
cv = Fscore_descriptive.iloc[2, :]/Fscore_descriptive.iloc[1, :] * 100
cv.name = 'cv %'
Fscore_descriptive = Fscore_descriptive.append(cv, ignore_index=False)
Fscore_descriptive = Fscore_descriptive.round(2)
Fscore_descriptive.drop(Fscore_descriptive.index[4:7], inplace=True)



# Make a table figure
fig, ax = plt.subplots(figsize=(14, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, Fscore_descriptive, loc='center')  # where df is your data frame
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(8) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1) # change size table


plt.savefig('./fig/summary_fscore.png', dpi=500)

# 3)D) Manage Outliers in Fscore
GB_zscore = (template_Fscore - template_Fscore.mean())/template_Fscore.std(ddof=0)
outliers_GB = pd.DataFrame()
for test in template_Fscore:
    loc = columns_Fscore.index(test)
    GB_zscore[test] = GB_zscore.iloc[:, loc]
    for i in range(len(GB_zscore[test])):
        if np.abs(GB_zscore.iloc[i, loc]) >= threshold:
            indice = i
            app_df = {'Player name': [name_GB[loc]],
                      'index': [loc],
                      'test': [test],
                      'Value zscore': [GB_zscore.iloc[i, loc]],
                      'Value abs': [template_Fscore.iloc[i, loc]]
                      }
            outliers_GB = outliers_GB.append(pd.DataFrame(data=app_df), ignore_index=True)

if outliers_GB.empty:
    print('There is no outliers in the Fscore dataframe')
else:
    for i in range(len(outliers_GB)):
        test_i = columns_Fscore.index(outliers_GB.iloc[i, 2])
        row_i = outliers_GB.iloc[i, 1]
        GB_zscore.iloc[row_i, test_i] = np.nan
        template_Fscore.iloc[row_i, test_i] = np.nan


# 3)E) Manage Boxplot for Fscore data Normalize
plt.figure(figsize=(12, 7))
melt_GB = GB_zscore.melt()
sns.boxplot(x="value",
            y="variable",
            data=melt_GB,
            color='grey'
            )
plt.title("Normalized Fscore boxplot of Groinbar testing", loc="center", fontsize=16)
plt.xlabel("Zscore")
plt.ylabel("Position")
plt.xlim((-4, 4))

# Save the Boxplot
plt.savefig('./fig/BP_GB_Fscore.png', dpi=500)

# 3)E)distribution imbalance not normalize
columns_imbalance = list(template_imb)
imb_distribution, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
imb_distribution.text(0.06, 0.5, '% of the data', va='center', rotation='vertical')
imb_distribution.text(0.5, 0.03, '% of imbalace', va='center', rotation='horizontal')
imb_distribution.suptitle('Distribution of Imbalance between legs side (%)', fontsize=16)
plt.xlim((-100, 100))
for test in template_imb:
    index = columns_imbalance.index(test)
    template_imb[test].dropna(axis=0, inplace=True)
    if index > 3:
        axx = 4
        axy = 2
    elif index > 1 and index < 4:
        axx = 3
        axy = 1
    elif index < 2:
        axx = 1
        axy = 0
    sns.distplot(
        template_imb[test],
        rug=True,
        fit=norm,
        kde_kws={"label": "KDE"},
        fit_kws={"color": "r", "lw": 1, "alpha": 0.6, "label": "Norm"},
        norm_hist=False,
        ax=axs[index - axx, axy]
    )
    axs[index - axx, axy].legend(loc='best')
    axs[index - axx, axy].set_xlabel("")
    axs[index - axx, axy].set_title(test)

#save the figure
plt.savefig('./fig/distribution_GBimb_.png', dpi=500)


# distribution imbalance normalize
Norm_imb_distribution, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
Norm_imb_distribution.text(0.06, 0.5, 'Density of probability', va='center', rotation='vertical')
Norm_imb_distribution.text(0.5, 0.03, 'Zscore', va='center', rotation='horizontal')
Norm_imb_distribution.suptitle('Normalized distribution of imbalance between legs side (%)', fontsize=16)
plt.xlim((-4, 4))
for test in template_imb:
    index = columns_imbalance.index(test)
    template_imb[test].dropna(axis=0, inplace=True)
    standardised = stats.zscore(template_imb[test])
    if index > 3:
        axx = 4
        axy = 2
    elif index > 1 and index < 4:
        axx = 3
        axy = 1
    elif index < 2:
        axx = 1
        axy = 0
    sns.distplot(
        standardised,
        rug=True,
        fit=norm,
        kde_kws={"label": "KDE"},
        fit_kws={"color": "r", "lw": 1, "alpha": 0.6, "label": "Norm"},
        norm_hist=False,
        ax=axs[index - axx, axy]
    )
    axs[index - axx, axy].legend(loc='best')
    axs[index - axx, axy].set_title(test)

#save the figure
plt.savefig('./fig/distribution_GBimb_norm.png', dpi=500)

# Manage outliers before doing BoxPlot
GBimb_zscore = (template_imb - template_imb.mean()) / template_imb.std(ddof=0)
outliers_GBimb = pd.DataFrame()
for test in template_imb:
    loc = columns_imbalance.index(test)
    GBimb_zscore[test] = GBimb_zscore.iloc[:, loc]
    for i in range(len(GBimb_zscore[test])):
        if np.abs(GBimb_zscore.iloc[i, loc]) >= threshold:
            indice = i
            app_df = {'Player name': [name_GB[i]],
                      'index': [i],
                      'test': [test],
                      'Value zscore': [GBimb_zscore.iloc[i, loc]],
                      'Value abs': [template_imb.iloc[i, loc]]
                      }
            outliers_GBimb = outliers_GBimb.append(pd.DataFrame(data=app_df), ignore_index=True)

if outliers_GBimb.empty:
    print('There is no outliers in the imb dataframe')
else:
    for i in range(len(outliers_GBimb)):
        test_i = columns_imbalance.index(outliers_GBimb.iloc[i, 2])
        row_i = outliers_GBimb.iloc[i, 1]
        GBimb_zscore.iloc[row_i, test_i] = np.nan
        template_imb.iloc[row_i, test_i] = np.nan

# 3)F) Boxplot of imbalance Normalize
plt.figure(figsize=(12, 7))
melt_GBimb = GBimb_zscore.melt()
sns.boxplot(x="value",
            y="variable",
            data=melt_GBimb,
            color='grey'
            )
plt.title("Normalized boxplot of Groinbar imbalance", loc="center", fontsize=16)
plt.xlabel("Zscore")
plt.ylabel("Position")
plt.xlim((-4, 4))

# Save the Boxplot
plt.savefig('./fig/BP_GBimb_norm.png', dpi=500)

#3) Descriptive statistic for imbalance
imb_descriptive = template_imb.describe()
imb_descriptive = imb_descriptive.round(2)
imb_descriptive.drop(imb_descriptive.index[4:7], inplace=True)



# Make a table figure
fig, ax = plt.subplots(figsize=(14, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, imb_descriptive, loc='center')  # where df is your data frame
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(8) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1) # change size table

# Save the descriptive table
plt.savefig('./fig/summary_imb.png', dpi=500)























# Make simple correlation
corr_table = off_ice.merge(on_ice.drop(columns=on_ice_todel), how='inner', on=['Name'])
pearson_table = corr_table.corr()
sns.heatmap(np.abs(pearson_table))

template_Fscore['Name'] = template['Name']
template_imb['Name'] = template['Name']
GB_all = template_Fscore.merge(template_imb, how='inner', on=['Name'])
corr_table_wGB = GB_all.merge(on_ice.drop(columns=on_ice_todel), how='inner', on=['Name'])
corr_table_wGB = corr_table_wGB[
    ['Name',
    'AD',
    'AB',
    'IR',
    'ER',
    'Flexion',
    'Extension',
    'imb_AD',
    'imb_AB',
    'imb_IR',
    'imb_ER',
    'imb_Flexion',
    'imb_Extension',
    '30M Forward',
    '30F-SPLIT 1 (5M)',
    '30F-SPLIT 2 (25M)',
    '30M Backward',
    '30B-SPLIT1 (5M)',
    '30B-SPLIT 2 (25M)',
    'Reaction',
    'Weave Agility',
    'Transition Agility']
]
pearson_table_wGB = corr_table_wGB.corr(method='pearson')
sns.heatmap(np.abs(pearson_table_wGB), annot=True)

corr_table_wGB_Norm = corr_table_wGB.merge(player_weight, how='inner', on=['Name'])
corr_table_wGB_Norm[['AD', 'AB', 'IR', 'ER', 'Flexion', 'Extension']] = corr_table_wGB_Norm[
    [
        'AD',
        'AB',
        'IR',
        'ER',
        'Flexion',
        'Extension'
    ]
].div(corr_table_wGB_Norm.Weight, axis=0)
corr_table_wGB_Norm.drop(
    columns=[
        'Weight'
    ],
    inplace=True
)
pearson_table_wGB_Norm = corr_table_wGB_Norm.corr(method='pearson')
sns.heatmap(np.abs(pearson_table_wGB_Norm))

corr_table_wGB_Norm.rename(
    columns={
        'AD':'AD_Norm',
        'AB':'AB_Norm',
        'IR':'IR_Norm',
        'ER':'ER_Norm',
        'Flexion': 'Flex_Norm',
        'Extension': 'Ext_Norm'
    },
    inplace=True
)




corr_GB  = corr_table_wGB.iloc[:, 0:7].merge(corr_table_wGB_Norm.iloc[:, 0:7], how='inner', on=['Name'])
pearson_table_GB = corr_GB.corr(method='pearson')
sns.heatmap(pearson_table_GB, annot=True)

corr, ax = plt.subplots(figsize=(12, 10))
tata = pearson_table_wGB_Norm.drop(
    index=corr_table_wGB.columns[1:13].tolist(),
    columns=columns_on_ice_nopuck
)
sns.heatmap(np.abs(tata), vmin=0, vmax=1, annot=True,linewidths=0.5, cmap="Greens")
plt.title('Pearson correlation between OKH on-ice testing \n and GroinBar testing', Fontsize=14)
plt.xlabel('GroinBar testing', Fontsize=14)
plt.ylabel('On-ice', Fontsize=14)
plt.savefig('./fig/pears_GB_ICE.png', dpi=500)


corr, ax = plt.subplots(figsize=(12, 10))
toto = pearson_table.drop(
    index=columns_off_ice,
    columns = columns_on_ice_nopuck,
)
sns.heatmap(np.abs(toto), vmin=0, vmax=1, annot=True, linewidths=0.5, cmap="Greens")
plt.title('Pearson correlation between OKH off-ice testing \n and OKH on-ice testing', Fontsize=14)
plt.xlabel('Off-ice', Fontsize=14)
plt.ylabel('On-ice', Fontsize=14)
plt.savefig('./fig/pears_OKH_ICE.png', dpi=500)

corr, ax = plt.subplots(figsize=(8.5, 6.5))
titi = pearson_table_GB.drop(
    index=pearson_table_GB.columns[0:6].tolist(),
    columns=pearson_table_GB.columns[6:].tolist(),
)
sns.heatmap(np.abs(titi), vmin=0, vmax=1, annot=True, linewidths=0.5, cmap="Greens")
plt.title('Pearson correlation between Groinbar testing \n Normalized VS not Normalized by weight of athlete', Fontsize=14)
plt.savefig('./fig/pears_GB_GBNorm.png', dpi=500)



#Take more variable in on-ice sprint
on_ice_forward = on_ice.iloc[:, 0:4]
on_ice_backward = on_ice.iloc[:, [0, 4, 5, 6]]

on_ice_forward['F_5_25'] = on_ice_forward['30F-SPLIT 2 (25M)'] - on_ice_forward['30F-SPLIT 1 (5M)']
on_ice_forward['F_25_30'] = on_ice_forward['30M Forward'] - on_ice_forward['30F-SPLIT 2 (25M)']
on_ice_forward['VF_5'] = 5/on_ice_forward['30F-SPLIT 1 (5M)']
on_ice_forward['VF_0_25'] = 25/on_ice_forward['30F-SPLIT 2 (25M)']
on_ice_forward['VF_5_25'] = 20/on_ice_forward['F_5_25']
on_ice_forward['VF_0_30'] = 30/on_ice_forward['30M Forward']
on_ice_forward['VF_25_30'] = 5/on_ice_forward['F_25_30']
on_ice_forward.rename(
    columns={
        '30M Forward': 'F_0_30',
        '30F-SPLIT 1 (5M)': 'F_0_5',
        '30F-SPLIT 2 (25M)': 'F_0_25',
    },
    inplace=True
)
on_ice_forward = on_ice_forward[
    [
        'Name',
        'F_0_5',
        'F_5_25',
        'F_0_25',
        'F_25_30',
        'F_0_30',
        'VF_5',
        'VF_0_25',
        'VF_5_25',
        'VF_0_30',
        'VF_25_30'
    ]
]

columns_on_ice_forward_time = on_ice_forward.columns[1:6].tolist()
columns_on_ice_forward_speed = on_ice_forward.columns[6:].tolist()


on_ice_forward_dist_time, axs = plt.subplots(nrows=5,
                                 ncols=1,
                                 sharey=True,
                                 sharex=True,
                                 figsize=(8, 7))
on_ice_forward_dist_time.subplots_adjust(hspace=0.4, wspace=0.4)
plt.xlim((0 , 6))
on_ice_forward_dist_time.text(0.08, 0.5, 'Density of probalility', va='center', rotation='vertical')
on_ice_forward_dist_time.text(0.5, 0.06, 'Time (sec)', va='center', rotation='horizontal')
on_ice_forward_dist_time.subplots_adjust(hspace=0.4, wspace=0.4)
on_ice_forward_dist_time.suptitle('Distribution of On-ice Forward Sprint Time (sec)', fontsize=16)


for test in columns_on_ice_forward_time:
    i = columns_on_ice_forward_time.index(test)
    sns.distplot(
        on_ice_forward[test],
        rug=True,
        fit=norm,
        kde_kws={"label": "KDE"},
        fit_kws={"color": "r", "lw": 1, "alpha": 0.6, "label": "Norm"},
        norm_hist=False,
        ax=axs[i],
    )
    axs[i].set_title(test)
    axs[i].set_xlabel("")
    axs[i].legend(loc='best')

on_ice_forward_dist_speed, axs = plt.subplots(nrows=5,
                                 ncols=1,
                                 sharey=True,
                                 sharex=True,
                                 figsize=(8, 7))
on_ice_forward_dist_speed.subplots_adjust(hspace=0.4, wspace=0.4)
plt.xlim((0, 13))
on_ice_forward_dist_speed.text(0.08, 0.5, 'Density of probalility', va='center', rotation='vertical')
on_ice_forward_dist_speed.text(0.5, 0.06, 'Speed (m/s)', va='center', rotation='horizontal')
on_ice_forward_dist_speed.suptitle('Distribution of On-ice Forward Sprint Speed (m/s)', fontsize=16)


for test in columns_on_ice_forward_speed:
    i = columns_on_ice_forward_speed.index(test)
    sns.distplot(
        on_ice_forward[test],
        rug=True,
        fit=norm,
        kde_kws={"label": "KDE"},
        fit_kws={"color": "r", "lw": 1, "alpha": 0.6, "label": "Norm"},
        norm_hist=False,
        ax=axs[i],
    )
    axs[i].set_title(test)
    axs[i].set_xlabel("")
    axs[i].legend(loc='best')

corr_OIF_GB_OFFI = on_ice_forward.merge(
    corr_GB.iloc[:, :7],
    how='inner',
    on=['Name']
)
corr_OIF_GB_OFFI = corr_OIF_GB_OFFI.merge(
    corr_table[['Name', 'Broad Jump', 'Pro Agility Left', 'Pro Agility Right', 'Vertical Jump', '20 M Sprint']],
    how='inner',
    on=['Name']
)
pearson_table_OIF_GB_OFFI = corr_OIF_GB_OFFI.corr(method='pearson')

corr, ax = plt.subplots(figsize=(10, 8))

tete = pearson_table_OIF_GB_OFFI.drop(
    index=pearson_table_OIF_GB_OFFI.columns[10:].tolist(),
    columns=pearson_table_OIF_GB_OFFI.columns[:10].tolist()
)
sns.heatmap(np.abs(tete), vmin=0, vmax=1, annot=True, linewidths=0.5, cmap="Greens")
plt.title('Pearson correlation between  off-ice testing \n and OKH on-ice forward sprint testing', Fontsize=14)
plt.tight_layout()
plt.xlabel('Off-ice', Fontsize=14)
plt.ylabel('On-ice', Fontsize=14)
plt.savefig('./fig/pears_OFFI_ICEF.png', dpi=500)

lol = stats.pearsonr(corr_OIF_GB_OFFI['F_0_30'], corr_OIF_GB_OFFI['20 M Sprint'])