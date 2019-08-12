# LHJMQ GroinBar analysis
# Import the package
import numpy as np
from scipy import signal
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set(style='whitegrid', context='paper')

os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/')
# Path creation and data collection
path = '.'
extension = 'csv'
os.chdir(path)
result = glob.glob('GroinBar/*.{}'.format(extension))

template = pd.read_excel('GroinBar/template.xlsx', index_col='Name')
template2try = pd.read_excel('GroinBar/template2try.xlsx', index_col='Name')

WINDOW_SIZE = 25
Sample_Rate = 50

# data collection
for test in result:
    df = pd.read_csv(test)
    name = df.iloc[2][0]
    test_name = df.iloc[2][5]
    print(f'participant name : {name}')
    print(f'\ttest position : {test_name}')
    tps = df.iloc[6:, 0].astype(float)
    force = pd.DataFrame(df.iloc[6:, 1:5]).astype(
        float)  # 'Left(squeeze)', 'Left(pull)', 'Right(squeeze)', 'Right(pull)'
    force = np.abs(force).reset_index().drop(columns='index')

    # Remove random peak

    # Filtering

    # design butterworth filter
    b, a = signal.butter(2, 10 / Sample_Rate, 'low', fs=Sample_Rate)
    filtered_signal = signal.filtfilt(b, a, force, axis=0)
    filtered_signal = pd.DataFrame(filtered_signal)

    #  smoothing data
    # smooth_force = filtered_signal.rolling(window=WINDOW_SIZE).mean()
    # rolling_median = filtered_signal.rolling(window=WINDOW_SIZE).median()
    # rolling RMS

    # Only take concerning columns
    if test_name == 'Hip AD/AB' or test_name == 'Hip IR/ER':
        force = filtered_signal.iloc[0:, 0:]
    else:
        force = filtered_signal.iloc[0:, [1, 3]]
    force.columns = range(force.shape[1])

    # Cut the try in 2 try or the best 2 try
    if test_name == 'Hip IR/ER':
        peak = force > 10
    else:
        if name == 'Anthony Bédard' and test_name == 'Hip Extension':
            peak = force > 70
        else:
            peak = force > 50
    peak = peak * 1
    peaksdiff = peak.diff(axis=0)

    for k in peak.columns:
        ascend = peaksdiff.index[peaksdiff[k] == 1].tolist()
        descend = peaksdiff.index[peaksdiff[k] == -1].tolist()
        maxrep = []
        zero = 0
        end = force.index[-1]
        if len(ascend) < len(descend):
            ascend = [zero] + ascend
        if len(ascend) > len(descend):
            descend = descend + [end]

        for idx, val in enumerate(ascend):
            if idx + 1 in enumerate(ascend):
                if ascend[idx + 1] - descend[idx] < Sample_Rate / 2:
                    ascend.remove(ascend[idx + 1])
                    descend.remove(descend[idx])
            maxperrep = force.iloc[ascend[idx]:descend[idx], k].max()
            maxrep.append(maxperrep)
            if idx == 2:
                maxrep.remove(min(maxrep))

            # generate the template_2try with the two try for each side and position
        if not maxrep:
            print(f'{name} in {test_name} for {k} column was not a rep')

        else:
            for idx in enumerate(maxrep):
                if test_name == 'Hip AD/AB':
                    if k == 0:
                        template2try.at[name, 'Hip AB left 1'] = maxrep[0]
                        template2try.at[name, 'Hip AB left 2'] = maxrep[1]
                    if k == 1:
                        template2try.at[name, 'Hip AD left 1'] = maxrep[0]
                        template2try.at[name, 'Hip AD left 2'] = maxrep[1]
                    if k == 2:
                        template2try.at[name, 'Hip AB right 1'] = maxrep[0]
                        template2try.at[name, 'Hip AB right 2'] = maxrep[1]
                    if k == 3:
                        template2try.at[name, 'Hip AD right 1'] = maxrep[0]
                        template2try.at[name, 'Hip AD right 2'] = maxrep[1]
                elif test_name == 'Hip IR/ER':
                    if k == 0:
                        template2try.at[name, 'Hip IR left 1'] = maxrep[0]
                        template2try.at[name, 'Hip IR left 2'] = maxrep[1]
                    if k == 1:
                        template2try.at[name, 'Hip ER left 1'] = maxrep[0]
                        template2try.at[name, 'Hip ER left 2'] = maxrep[1]
                    if k == 2:
                        template2try.at[name, 'Hip IR right 1'] = maxrep[0]
                        template2try.at[name, 'Hip IR right 2'] = maxrep[1]
                    if k == 3:
                        template2try.at[name, 'Hip ER right 1'] = maxrep[0]
                        template2try.at[name, 'Hip ER right 2'] = maxrep[1]
                elif test_name == 'Hip Flexion':
                    if k == 0:
                        template2try.at[name, 'Hip Flexion left 1'] = maxrep[0]
                        template2try.at[name, 'Hip Flexion left 2'] = maxrep[1]
                    if k == 1:
                        template2try.at[name, 'Hip Flexion right 1'] = maxrep[0]
                        template2try.at[name, 'Hip Flexion right 2'] = maxrep[1]
                elif test_name == 'Hip Extension':
                    if k == 0:
                        template2try.at[name, 'Hip Extension left 1'] = maxrep[0]
                        template2try.at[name, 'Hip Extension left 2'] = maxrep[1]
                    if k == 1:
                        template2try.at[name, 'Hip Extension right 1'] = maxrep[0]
                        template2try.at[name, 'Hip Extension right 2'] = maxrep[1]

            # Build the final Dataframe
    for column in force.columns:
        max_force = force.max()

    if test_name == 'Hip AD/AB':
        template.at[name, 'Hip AB left'] = max_force[0]
        template.at[name, 'Hip AD left'] = max_force[1]
        template.at[name, 'Hip AB right'] = max_force[2]
        template.at[name, 'Hip AD right'] = max_force[3]

    elif test_name == 'Hip IR/ER':
        template.at[name, 'Hip IR left'] = max_force[0]
        template.at[name, 'Hip ER left'] = max_force[1]
        template.at[name, 'Hip IR right'] = max_force[2]
        template.at[name, 'Hip ER right'] = max_force[3]

    elif test_name == 'Hip Flexion':
        template.at[name, 'Hip Flexion left'] = max_force[0]
        template.at[name, 'Hip Flexion right'] = max_force[1]

    elif test_name == 'Hip Extension':
        if max_force[0] > 50:
            template.at[name, 'Hip Extension left'] = max_force[0]
        if max_force[1] > 50:
            template.at[name, 'Hip Extension right'] = max_force[1]


template.sort_index(
    inplace=True
)
template.to_pickle('template.pkl')

#consent = pd.read_excel('consent.xlsx', index_col='Name')
#template = pd.concat([template, antropo['Position'], consent], axis=1, sort=True)
#template = template[template.Position != 'GOALIE'].dropna(subset=['GB consent', 'OKH consent']).drop(columns=['Position', 'GB consent', 'OKH consent'])

# clean the template, erase Name with NaN in test
#template2try = template2try.dropna().sort_index()
#template_kg = template.iloc[0:, 0:]/9.81 #Mets les forces seulement en Kg

#template2try.to_pickle('template2try.pkl')
#template_kg.to_pickle('template_kg.pkl')

#listname = template.dropna().reset_index()
#listname = listname['index'].values.tolist()

#with open('listname', 'wb') as f:
#    pickle.dump(listname, f)


#template_with_weitgh_height_position = pd.merge(antropo, template, on='Name', how='inner').sort_values('Name', ascending='false').reset_index(drop=True)
#template_with_weitgh_height_position.iloc[:, 4:] = template_with_weitgh_height_position.iloc[:, 4:].div(template_with_weitgh_height_position.Weight, axis=0)

# template.to_excel('templateAll.xlsx', 'All', na_rep='NaN')
# template.dropna(0, 'all').to_excel('templateSubAll.xlsx', 'SubAll', na_rep='NaN')
# template.dropna().to_excel('templateAllTest.xlsx', 'Alltest')

# generate relative force, approximation of both leg force, ratio ago/antago, imbalance dominant VS non dominant legs

# generate summary graph, box plot of all variable, all position normalized force, ratio ago/antago, imbalance dominant/nondominant
# percentiles = np.array([25, 50, 75])
# perc_add, perc_abd, perc_rotint, perc_rotext, perc_flex, perc_ext = np.percentile(template.iloc[:, ])
