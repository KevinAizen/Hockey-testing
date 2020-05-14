# This code will find out if some of the features or target variable are normaly distributed.
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro, zscore
import numpy as np

os.chdir("./results")

test = pd.read_pickle("./template/all_data/test_relative_GB.pkl")
test.drop(columns=['Position'], inplace=True)

test.dropna(inplace=True)

#Zscore test
zscore_test = zscore(test)
zscore_test = pd.DataFrame(zscore_test, columns=test.columns)
zscore_test.hist()

#Only imb_
imbalance_df = test.filter(regex='imb_')
imbalance_df.hist()

zcore_imbalance_df = zscore(imbalance_df)
zcore_imbalance_df = pd.DataFrame(zcore_imbalance_df, columns=imbalance_df.columns)
zcore_imbalance_df.hist()

log_imbalance_df = np.log(imbalance_df)
log_imbalance_df.hist()

test.hist(bins=10, figsize=(20,15))



for col in test:
    stat, p = shapiro(zscore_test[col])
    print('variable: %s, Statistics=%.3f, p=%.3f' % (col, stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    print(f'{"-" * 30}\n')