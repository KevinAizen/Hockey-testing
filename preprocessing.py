import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import savetxt
from function import Confidence_interval_95

path = os.getcwd()

sns.set(style="whitegrid", context="paper")

# Output will be stable
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

import altair as alt
alt.data_transformers.enable("json")


## Get the data
### 1) Features X

os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/results/')

X = pd.read_excel('X.xlsx')
# X.drop(
#     columns='Position',
#     inplace=True,
# )
X = X.iloc[:, 1:]
y = pd.read_excel('y.xlsx')

GroinBar_all = ["AD",  "imb_AD", "AB", "imb_AB", "IR", "imb_IR", "ER", "imb_ER", "Flexion", "imb_Flexion", "Extension", "imb_Extension"]
GroinBar_fscore = ["AD", "AB", "IR", "ER", "Flexion", "Extension"]
GroinBar_imb = ["imb_AD", "imb_AB", "imb_IR", "imb_ER", "imb_Flexion", "imb_Extension"]

#GroinBar_best_corr = [ "AD", "AB", "Flexion"]

X_var1 = [ "Vertical Jump", "20 M Sprint", "Wingate w/kg", "AD", "AB", "Flexion"]
X['Height'] = X['Height'].div(100, axis=0)
X[GroinBar_fscore] = X[GroinBar_fscore].div(X['Weight'], axis=0)


X = X.rename(
    columns={"Broad Jump": "BJ", "Vertical Jump": "VJ", "20 M Sprint": "20M", "Wingate w/kg": "Wingate",
             "Flexion": "FLEX", "Extension": "EXT", "imb_Flexion": "imb_FLEX", "imb_Extension": "imb_EXT"}
)
#X = X[X_var1]

# Not taking imbalance for features
GroinBar_imb = ["imb_AD", "imb_AB", "imb_IR", "imb_ER", "imb_FLEX", "imb_EXT"]
# X = X.drop(GroinBar_imb, axis=1)




#X.drop(columns=['BJ', 'Height', 'Weight', 'VJ', 'Wingate', '20M'], inplace=True)
#X.dropna(axis=1, inplace=True)
#inds = pd.isnull(X).any(1).nonzero()[0]



X.dropna(inplace=True)
X.reset_index(drop=True, inplace=True)
#X.reset_index(drop=True, inplace=True)
print(X.shape)

X_descriptive = X.describe()
X_descriptive = Confidence_interval_95(describe_df=X_descriptive)

X_columns_name = X.columns.tolist()
X_index_list = list(X.index)

# 2) target y
y = y.iloc[X_index_list, 1:]
y.reset_index(drop=True, inplace=True)
y.shape
y.drop(
    columns=y.filter(like='V', axis=1).columns.values,
    inplace=True
)
target = y.columns.tolist()

y_descriptive = y.describe()
y_descriptive = Confidence_interval_95(describe_df=y_descriptive)



# 3) Concatenate X, y in one df prior to splitting dataset
test = pd.concat([X, y], join='inner', axis=1)
test.reset_index(inplace=True)


# 4) Splitting Data set onto Training, Strat_Training, testing, Strat Training dataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Better generalization between training data set and training set, we will have the same proportion of F and D
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
for train_index, test_index in split.split(test, test['Position']):
    strat_train_set = test.loc[train_index]
    strat_test_set = test.loc[test_index]

train_set, test_set = train_test_split(test, test_size=0.2, random_state=RANDOM_SEED)

print(strat_test_set['Position'].value_counts()/len(strat_test_set))

print(test_set['Position'].value_counts()/len(test_set))

X_train_set = strat_train_set[X_columns_name]
y_train_set = strat_train_set[target]
X_test_set = strat_test_set[X_columns_name]
y_test_set = strat_test_set[target]


# Drop columns that i don't want to test
X_train_set = X_train_set.drop(columns=['Position'])
X_columns_name = X_train_set.columns.tolist()



# 5)preprocessing the data using pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

#X_train_set_num = X_train_set.drop("Position", axis=1)
X_train_set_num = X_train_set.copy()
num_attribs = list(X_train_set_num.columns)
#cat_attribs = ["Position"]

# num_pipeline = Pipeline([
#     #('imputer', SimpleImputer(strategy='median')),
#     #('std_scaler', StandardScaler())
# ])

# full_pipeline = ColumnTransformer([
#     #("cat", OrdinalEncoder(), cat_attribs),
#     ("num", num_pipeline, num_attribs)
# ])

#X_train_set_prepared = full_pipeline.fit_transform(X_train_set_num)
X_train_set_prepared = X_train_set_num

# # Saving training and testing set
# test.to_pickle("./template/all_data/test_relative_GB_no_nan.pkl")
# y_train_set.to_pickle("./template/y_train_set/y_train_set_noV_no_nan.pkl")
# X_test_set.to_pickle("./template/X_test_set/X_test_scale_relative_GB_no_nan.pkl")
# y_test_set.to_pickle("./template/y_test_set/y_test_set_noV_no_nan.pkl")

# with open('./template/X_columns_name/X_columns_name.pkl', 'wb') as f:
#     pickle.dump(X_columns_name, f)
#
# os.chdir('./template/X_train_set/')
# savetxt('X_train_set_prepared_not_scale_relative_GB_no_nan_imb.csv', X_train_set_prepared, delimiter=',')

# #Save descriptive table
# os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/results/')
# X_descriptive.to_pickle("./template/descriptive_stats/X_descriptive_relative_GB_no_nan.pkl")
# with pd.ExcelWriter('./template/descriptive_stats/stats_no_nan.xlsx') as writer:
#     X_descriptive.to_excel(writer, sheet_name='descriptive_stats_off_ice')
#     y_descriptive.to_excel(writer, sheet_name='descriptive_stats_on_ice')