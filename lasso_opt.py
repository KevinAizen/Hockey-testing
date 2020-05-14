import pandas as pd
import pickle
import os
import numpy as np
from numpy import loadtxt
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsCV, lasso_path, LassoCV
from sklearn.model_selection import cross_validate
from function import display_scores
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
from function import model_result

path = os.getcwd()

os.chdir(path + "/results/template/X_train_set/")
X_train_set_prepared = loadtxt("X_train_set_prepared_scale_relative_GB_no_nan.csv", delimiter=',')

os.chdir(path + "/results/template/y_train_set/")
y_train_set = pd.read_pickle("y_train_set_noV_no_nan.pkl")

os.chdir(path + "/results/template/X_columns_name/")
X_columns_name = pd.read_pickle("X_columns_name.pkl")


def lasso_regression(target, predictors):
    y_pred = pd.DataFrame(columns=target.columns.values)

    scoring = {'nmse': 'neg_root_mean_squared_error',
               'mape': 'neg_mean_relative_error',
               'r2': 'r2'
               }

    lassoreg = {}
    ret = {}
    lasso_reg_rmse_scores = {}
    lasso_reg_mape_scores = {}
    alpha = {}
    path_lasso = {}
    mse_CV = {}
    lasso_reg_r2_scores = {}

    # Fit the model
    for itest, test in enumerate(target):
        lassoreg[test] = LassoCV(cv=5)
        lassoreg[test].fit(predictors, target[test])
        alpha[test] = lassoreg[test].alpha_

        path_lasso[test] = lassoreg[test].path(predictors, target[test])
        mse_CV[test] = lassoreg[test].mse_path_

        scores_lasso_reg = cross_validate(lassoreg[test], predictors, target[test],
                                          scoring=scoring, cv=10)

        # return the result in pre-defined format
        scores_lasso_reg['test_nmse'] = -scores_lasso_reg['test_nmse']
        scores_lasso_reg['test_mape'] = np.abs(scores_lasso_reg['test_mape'])
        scores_lasso_reg['test_r2'] = np.abs(scores_lasso_reg['test_r2'])


        lasso_reg_rmse_scores[test] = scores_lasso_reg['test_nmse']
        lasso_reg_mape_scores[test] = scores_lasso_reg['test_mape']
        lasso_reg_r2_scores[test] = scores_lasso_reg['test_r2']

        ret[test] = [lasso_reg_rmse_scores[test].mean(),
                     lasso_reg_rmse_scores[test].std(),
                     lasso_reg_mape_scores[test].mean(),
                     lasso_reg_mape_scores[test].std(),
                     lasso_reg_r2_scores[test].mean()]

        ret[test].extend([lassoreg[test].alpha_])
        ret[test].extend([lassoreg[test].intercept_])
        ret[test].extend(lassoreg[test].coef_)

    col = ['rmse', 'rmse_std', 'mape', 'mape_std', 'r2', 'alpha', 'intercept'] + X_columns_name
    ret = pd.DataFrame(ret).T
    ret.columns = col
    import pandas as pd
    import pickle
    import os
    import numpy as np
    from numpy import loadtxt
    from sklearn.linear_model import LinearRegression, Lasso, LassoLarsCV, lasso_path, LassoCV
    from sklearn.model_selection import cross_validate
    from function import display_scores
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
    from function import model_result

    path = os.getcwd()

    os.chdir(path + "/results/template/X_train_set/")
    X_train_set_prepared = loadtxt("X_train_set_prepared_scale_relative_GB_no_nan.csv", delimiter=',')

    os.chdir(path + "/results/template/y_train_set/")
    y_train_set = pd.read_pickle("y_train_set_noV_no_nan.pkl")

    os.chdir(path + "/results/template/X_columns_name/")
    X_columns_name = pd.read_pickle("X_columns_name.pkl")

    def lasso_regression(target, predictors):
        y_pred = pd.DataFrame(columns=target.columns.values)

        scoring = {'nmse': 'neg_root_mean_squared_error',
                   'mape': 'neg_mean_relative_error',
                   'r2': 'r2'
                   }

        lassoreg = {}
        ret = {}
        lasso_reg_rmse_scores = {}
        lasso_reg_mape_scores = {}
        alpha = {}
        path_lasso = {}
        mse_CV = {}
        lasso_reg_r2_scores = {}

        # Fit the model
        for itest, test in enumerate(target):
            lassoreg[test] = LassoCV(cv=5)
            lassoreg[test].fit(predictors, target[test])
            alpha[test] = lassoreg[test].alpha_

            path_lasso[test] = lassoreg[test].path(predictors, target[test])
            mse_CV[test] = lassoreg[test].mse_path_

            scores_lasso_reg = cross_validate(lassoreg[test], predictors, target[test],
                                              scoring=scoring, cv=10)

            # return the result in pre-defined format
            scores_lasso_reg['test_nmse'] = -scores_lasso_reg['test_nmse']
            scores_lasso_reg['test_mape'] = np.abs(scores_lasso_reg['test_mape'])
            scores_lasso_reg['test_r2'] = np.abs(scores_lasso_reg['test_r2'])

            lasso_reg_rmse_scores[test] = scores_lasso_reg['test_nmse']
            lasso_reg_mape_scores[test] = scores_lasso_reg['test_mape']
            lasso_reg_r2_scores[test] = scores_lasso_reg['test_r2']

            ret[test] = [lasso_reg_rmse_scores[test].mean(),
                         lasso_reg_rmse_scores[test].std(),
                         lasso_reg_mape_scores[test].mean(),
                         lasso_reg_mape_scores[test].std(),
                         lasso_reg_r2_scores[test].mean()]

            ret[test].extend([lassoreg[test].alpha_])
            ret[test].extend([lassoreg[test].intercept_])
            ret[test].extend(lassoreg[test].coef_)

        col = ['rmse', 'rmse_std', 'mape', 'mape_std', 'r2', 'alpha', 'intercept'] + X_columns_name
        ret = pd.DataFrame(ret).T
        ret.columns = col

        return ret, lassoreg, path_lasso, mse_CV, lasso_reg_r2_scores

    matrix_lasso, lassoreg, path_lasso, mse_CV, lasso_reg_r2_scores = lasso_regression(y_train_set,
                                                                                       X_train_set_prepared)

    coef_matrix_lasso = matrix_lasso.iloc[:, 6:]
    coef_matrix_lasso = coef_matrix_lasso.T
    feature_list = {}
    for test_name in coef_matrix_lasso:
        mask = coef_matrix_lasso[test_name] != 0
        feature_list[test_name] = coef_matrix_lasso.index[mask == True].tolist()

    # See the result of the prediction on the test set.
    from sklearn.preprocessing import StandardScaler
    # import test_set
    os.chdir(path + "/results/template/X_test_set/")
    X_test = pd.read_pickle("X_test_scale_relative_GB_no_nan.pkl")
    X_test.drop(columns=['Position'], inplace=True)
    X_test = StandardScaler().fit_transform(X_test)
    X_test = pd.DataFrame(X_test, columns=X_columns_name)

    os.chdir(path + "/results/template/y_test_set/")
    y_true = pd.read_pickle("y_test_set_noV_no_nan.pkl")
    y_true.reset_index(drop=True, inplace=True)

    lassoreg_results, array_results_rmse, array_results_mape = model_result(lassoreg, X_test, y_true)
    return ret, lassoreg, path_lasso, mse_CV, lasso_reg_r2_scores


matrix_lasso, lassoreg, path_lasso, mse_CV, lasso_reg_r2_scores = lasso_regression(y_train_set, X_train_set_prepared)

coef_matrix_lasso = matrix_lasso.iloc[:, 6:]
coef_matrix_lasso = coef_matrix_lasso.T
feature_list =  {}
for test_name in coef_matrix_lasso:
    mask = coef_matrix_lasso[test_name] != 0
    feature_list[test_name] = coef_matrix_lasso.index[mask == True].tolist()


# See the result of the prediction on the test set.
from sklearn.preprocessing import StandardScaler
# import test_set
os.chdir(path + "/results/template/X_test_set/")
X_test = pd.read_pickle("X_test_scale_relative_GB_no_nan.pkl")
X_test.drop(columns=['Position'], inplace=True)
X_test = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(X_test, columns=X_columns_name)


os.chdir(path + "/results/template/y_test_set/")
y_true = pd.read_pickle("y_test_set_noV_no_nan.pkl")
y_true.reset_index(drop=True, inplace=True)

lassoreg_results, array_results_rmse, array_results_mape  = model_result(lassoreg, X_test, y_true)

# #Save the model and results
# path = '/Users/kevinaizen/PycharmProjects/analysis_groinbar/results/'
# output = open(path+"model/final/Lasso_opt_scale_relativeGB_no_nan.pkl", "wb")
# pickle.dump(lassoreg, output)
# output.close()
#
# output = open(path+"test_set_result/article/Lasso_opt_array_results_rmse.pkl", "wb")
# pickle.dump(array_results_rmse, output)
# output.close()
#
# output = open(path+"test_set_result/article/Lasso_opt_array_results_mape.pkl", "wb")
# pickle.dump(array_results_mape, output)
# output.close()
#
# output = open(path+"test_set_result/article/Lasso_opt_results.pkl", "wb")
# pickle.dump(lassoreg_results, output)
# output.close()

