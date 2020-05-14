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

WINDOW_SIZE = 25
Sample_Rate = 50

sns.set(style='whitegrid', context='paper')

os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/')
# Path creation and data collection
path = '.'
extension = 'csv'
os.chdir(path)
result = glob.glob('GroinBar/*.{}'.format(extension))


GB_Max_2 = pd.read_excel('GB_template/template2try.xlsx', index_col='Name')
GB_Max_Best = pd.read_excel('GB_template/template.xlsx', index_col='Name')
GB_Md_2 = pd.read_excel('GB_template/template2try.xlsx', index_col='Name')
GB_Md_Best = pd.read_excel('GB_template/template.xlsx', index_col='Name')

# data collection
for test in result:
    if test == 'GroinBar/Zachary-L\'Heureux_24042019_i8525298.csv':
    #if test == 'GroinBar/Charles-Ã‰mile-Duciaume_24042019_i8525252.csv':
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
            b, a = signal.butter(2, 10, 'low', fs=Sample_Rate)
            force = signal.filtfilt(b, a, force, axis=0)
            force = pd.DataFrame(force)

            #  smoothing data
            # smooth_force = filtered_signal.rolling(window=WINDOW_SIZE).mean()
            # rolling_median = filtered_signal.rolling(window=WINDOW_SIZE).median()
            # rolling RMS

            # Only take concerning columns
            if test_name == 'Hip AD/AB' or test_name == 'Hip IR/ER':
                force = force.iloc[0:, 0:]
            else:
                force = force.iloc[0:, [1, 3]]
            force.columns = range(force.shape[1])

            # Cut the try in 2 try or the best 2 try
            if test_name == 'Hip IR/ER':
                if name == 'Anthony Bédard' or name =='Carter McCluskey':
                    peak = force > 10
                else:
                    peak = force > 40

            else:
                if name == 'Anthony Bédard' and test_name == 'Hip Extension':
                    peak = force > 70
                elif name == 'Jonathan Desrosiers' and test_name == 'Hip Extension':
                    peak = force > 100
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

                for idx in range(len(ascend)):
                    if idx + 1 in range(len(ascend)):
                        if ascend[idx + 1] - descend[idx] < Sample_Rate:
                            ascend.remove(ascend[idx + 1])
                            descend.remove(descend[idx])

                lenght = np.subtract(descend, ascend)
                gg = [i for i in range(len(lenght)) if lenght[i] > 1 * Sample_Rate]
                ascend = [ascend[i] for i in gg]
                descend = [descend[i] for i in gg]
                i = len(ascend)

                if i != 0:
                    for idx in range(len(ascend)):
                        # Take the max value but the more constant value
                        md = force.iloc[ascend[idx]:descend[idx], k].median()
                        #max = rep.rolling(window=25).max()
                        #min = rep.rolling(window=25).min()
                        #spread = max - min
                        #avg = rep.rolling(window=25).mean()
                        #findrealmax = pd.concat([avg, spread], axis=1)
                        #findrealmax = findrealmax[findrealmax.iloc[:, 1] < 50]
                        #maxperrep = findrealmax.iloc[:, 0].max()
                        maxrep.append(md)

                        # only take the 2 maximum value
                    if idx >= 2:
                        MAX = pd.DataFrame({'Value': maxrep})
                        MAX = MAX.sort_values(by=['Value'], ascending=False).iloc[:2]
                        # MAX = MAX[MAX['Value'] != MAX['Value'].min()]
                        # maxrep = MAX['Value'].tolist()

                    else:
                        MAX = pd.DataFrame({'Value': maxrep})

                    # generate the template_2try with the two try for each side and position
                if i == 0:
                    print(f'{name} in {test_name} for {k} column was not a rep')

                else:
                    for idx in enumerate(maxrep):
                        if test_name == 'Hip AD/AB':
                            if k == 0:
                                GB_Md_2.at[name, 'Hip AB left 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip AB left 2'] = MAX.iloc[1]
                            if k == 1:
                                GB_Md_2.at[name, 'Hip AD left 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip AD left 2'] = MAX.iloc[1]
                            if k == 2:
                                GB_Md_2.at[name, 'Hip AB right 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip AB right 2'] = MAX.iloc[1]
                            if k == 3:
                                GB_Md_2.at[name, 'Hip AD right 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip AD right 2'] = MAX.iloc[1]
                        elif test_name == 'Hip IR/ER':
                            if k == 0:
                                GB_Md_2.at[name, 'Hip IR left 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip IR left 2'] = MAX.iloc[1]
                            if k == 1:
                                GB_Md_2.at[name, 'Hip ER left 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip ER left 2'] = MAX.iloc[1]
                            if k == 2:
                                GB_Md_2.at[name, 'Hip IR right 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip IR right 2'] = MAX.iloc[1]
                            if k == 3:
                                GB_Md_2.at[name, 'Hip ER right 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip ER right 2'] = MAX.iloc[1]
                        elif test_name == 'Hip Flexion':
                            if k == 0:
                                GB_Md_2.at[name, 'Hip Flexion left 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip Flexion left 2'] = MAX.iloc[1]
                            if k == 1:
                                GB_Md_2.at[name, 'Hip Flexion right 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip Flexion right 2'] = MAX.iloc[1]
                        elif test_name == 'Hip Extension':
                            if k == 0:
                                GB_Md_2.at[name, 'Hip Extension left 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip Extension left 2'] = MAX.iloc[1]
                            if k == 1:
                                GB_Md_2.at[name, 'Hip Extension right 1'] = MAX.iloc[0]
                                GB_Md_2.at[name, 'Hip Extension right 2'] = MAX.iloc[1]

column2try = list(GB_Md_2)
# Build the final Dataframe with best try
for test in column2try:
    index = column2try.index(test)
    if test[-1] == '1':
        max2try = GB_Md_2.iloc[:, [index, index+1]].max(axis=1)

        if test[:-2] == 'Hip Extension right':
            GB_Md_Best['Hip Extension right'] = max2try
        if test[:-2] == 'Hip Extension left':
            GB_Md_Best['Hip Extension left'] = max2try

        elif test[:-2] == 'Hip AD right':
            GB_Md_Best['Hip AD right'] = max2try
        elif test[:-2] == 'Hip AD left':
            GB_Md_Best['Hip AD left'] = max2try

        elif test[:-2] == 'Hip AB right':
            GB_Md_Best['Hip AB right'] = max2try
        elif test[:-2] == 'Hip AB left':
            GB_Md_Best['Hip AB left'] = max2try

        elif test[:-2] == 'Hip IR right':
            GB_Md_Best['Hip IR right'] = max2try
        elif test[:-2] == 'Hip IR left':
            GB_Md_Best['Hip IR left'] = max2try

        elif test[:-2] == 'Hip ER right':
            GB_Md_Best['Hip ER right'] = max2try
        elif test[:-2] == 'Hip ER left':
            GB_Md_Best['Hip ER left'] = max2try

        elif test[:-2] == 'Hip Flexion right':
            GB_Md_Best['Hip Flexion right'] = max2try
        elif test[:-2] == 'Hip Flexion left':
            GB_Md_Best['Hip Flexion left'] = max2try

GB_Md_Best.sort_index(
    inplace=True
)
GB_Md_2.sort_index(
    inplace=True
)

os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/results')

with pd.ExcelWriter('GB_Md.xlsx') as writer:
    GB_Md_2.to_excel(writer, sheet_name='2Try')
    GB_Md_Best.to_excel(writer, sheet_name='BestTry')




#templatetest1.to_pickle('templatetest1.pkl')

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
