import pandas as pd
import pickle
import os
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from function import display_scores, backwardElimination
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

path = os.getcwd()

os.chdir(path+"/results/template/X_train_set/")
X_train_set_prepared = loadtxt("X_train_set_prepared_scale_relative_GB_no_nan.csv", delimiter=',')

os.chdir(path+"/results/template/y_train_set/")
y_train_set = pd.read_pickle("y_train_set_noV_no_nan.pkl")

os.chdir(path+"/results/template/X_columns_name/")
X_columns_name = pd.read_pickle("X_columns_name.pkl")

lin_reg = {}
lin_reg_rmse_scores = {}
lin_reg_std_rmse_score = {}
lin_reg_mape_scores = {}
lin_reg_std_mape_score = {}
features = {}

rmse_OLS = pd.DataFrame()
mape_OLS = pd.DataFrame()

scoring = {'nmse': 'neg_mean_squared_error',
           'mape': 'neg_mean_relative_error'
           }

for itarget, itarget_name in enumerate(y_train_set.columns):
    print(f"\t{itarget}. {itarget_name}\n")

    y_train = y_train_set[itarget_name]

    #X_modeled = backwardElimination(X_train_set_prepared, y_train, X_columns_name, 0.05)
    #features[itarget_name] = X_modeled.columns.tolist()
    X_modeled = pd.DataFrame(X_train_set_prepared, columns=X_columns_name)

    lin_reg[itarget_name] = LinearRegression()

    lin_reg[itarget_name].fit(X_modeled, y_train_set[itarget_name])

    scores_lin_reg = cross_validate(lin_reg[itarget_name], X_modeled, y_train,
                                     scoring=scoring, cv=10)

    scores_lin_reg['test_nmse'] = -scores_lin_reg['test_nmse']
    scores_lin_reg['test_mape'] = np.abs(scores_lin_reg['test_mape'])

    lin_reg_rmse_scores[itarget_name] = scores_lin_reg['test_nmse']
    lin_reg_std_rmse_score[itarget_name] = np.std(scores_lin_reg['test_nmse'])
    lin_reg_mape_scores[itarget_name] = scores_lin_reg['test_mape']
    lin_reg_std_mape_score[itarget_name] = np.std(scores_lin_reg['test_mape'])

    print(display_scores(lin_reg_rmse_scores[itarget_name], lin_reg_mape_scores[itarget_name]))
    print(f'{"-" * 15}\n')

# #Save model
# output = open("linreg_allstrat.pkl", "wb")
# pickle.dump(lin_reg, output)
# output.close()

#Make DF for model comparison fig


lin_reg_rmse_scores = pd.DataFrame.from_dict(lin_reg_rmse_scores, orient="columns")
mean_cv = lin_reg_rmse_scores.mean()
std_cv = lin_reg_rmse_scores.std()
lin_reg_rmse_scores = lin_reg_rmse_scores.append([mean_cv, std_cv], ignore_index=True)
lin_reg_rmse_scores.rename(index={10: 'mean', 11: 'std'}, inplace=True)

lin_reg_mape_scores = pd.DataFrame.from_dict(lin_reg_mape_scores, orient="columns")
mean_cv = lin_reg_mape_scores.mean()
std_cv = lin_reg_mape_scores.std()
lin_reg_mape_scores = lin_reg_mape_scores.append([mean_cv, std_cv], ignore_index=True)
lin_reg_mape_scores.rename(index={10: 'mean', 11: 'std'}, inplace=True)


#Save the model
path = '/Users/kevinaizen/PycharmProjects/analysis_groinbar/results'
output = open(path+"/model/lin_reg_scale_relativeGB_no_nan_10CV.pkl", "wb")
pickle.dump(lin_reg, output)
output.close()

# Get the result on the test_set
# load model
os.chdir(path+"/model/final/")
pkl_file = open('Lin_Reg_scale_relativeGB_no_nan_10CV.pkl', 'rb')
linreg = pickle.load(pkl_file)
pkl_file.close()

# Load testing set
# See the result of the prediction on the test set.
from function import model_result
from sklearn.preprocessing import StandardScaler
# import test_set
os.chdir(path + "/template/X_test_set/")
X_test = pd.read_pickle("X_test_scale_relative_GB_no_nan.pkl")
X_test.drop(columns=['Position'], inplace=True)
X_test = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(X_test, columns=X_columns_name)

os.chdir(path + "/template/y_test_set/")
y_true = pd.read_pickle("y_test_set_noV_no_nan.pkl")
y_true.reset_index(drop=True, inplace=True)

linreg_results, array_results_rmse, array_results_mape = model_result(lin_reg, X_test, y_true)

# # Save model test result
# output = open(path+"/test_set_result/article/Lin_reg_array_results_rmse.pkl", "wb")
# pickle.dump(array_results_rmse, output)
# output.close()
#
# output = open(path+"/test_set_result/article/Lin_reg_array_results_mape.pkl", "wb")
# pickle.dump(array_results_mape, output)
# output.close()
#
# output = open(path+"/test_set_result/article/Lin_reg_results.pkl", "wb")
# pickle.dump(linreg_results, output)
# output.close()

# #Save df
# cv_score_rmse = lin_reg_rmse_scores
# cv_score_rmse.to_pickle(path+"/results/cv/lin_reg/cv_score_rmse_scale_relativeGB_no_nan.pkl")
#
# cv_score_mape = lin_reg_mape_scores
# cv_score_mape.to_pickle(path+"/results/cv/lin_reg/cv_score_mape_scale_relativeGB_no_nan.pkl")
