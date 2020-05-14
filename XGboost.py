import pandas as pd
import pickle
import os
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import cross_validate
import xgboost as xgb
from skopt import BayesSearchCV, dump, load
import altair as alt
from itertools import combinations
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pymc3 as pm
from function import display_scores, model_result

alt.data_transformers.enable("json")

RANDOM_SEED = 42

path = os.getcwd()

os.chdir(path+"/results")
# Load columns name
X_columns_name = pd.read_pickle(path+"/results/template/X_columns_name/X_columns_name.pkl")
X_columns_name_Guo = pd.read_pickle(path+"/results/template/X_columns_name/X_columns_name_Guo.pkl")
# Load Data set
X_train = loadtxt("./template/X_train_set/X_train_set_prepared_not_scale_relative_GB_no_nan.csv", delimiter=',')
X_train = pd.DataFrame(X_train, columns=X_columns_name)
y_train = pd.read_pickle("./template/y_train_set/y_train_set_noV_no_nan.pkl")
X_test = pd.read_pickle("./template/X_test_set/X_test_scale_relative_GB_no_nan.pkl")
X_test.drop(columns=['Position'], inplace=True)
y_test = pd.read_pickle("./template/y_test_set/y_test_set_noV_no_nan.pkl")


class BayesSearch:
    def __init__(self, model, search_spaces, n_iter, export_path):
        self.export_path = export_path
        self.bayes_cv_tuner = BayesSearchCV(
            model,
            search_spaces,
            cv=5,
            n_jobs=-1,
            n_iter=n_iter,
            verbose=0,
            refit=True,
            random_state=RANDOM_SEED,
        )

    def fit(self, X, y):
        self.bayes_cv_tuner.fit(X, y, callback=self.print_status)
        self.export_results()

    def export_results(self):
        pd.DataFrame(self.bayes_cv_tuner.cv_results_).to_csv(
            f"{self.export_path}_cv_results.csv"
        )
        pd.Series(self.bayes_cv_tuner.best_params_).to_json(
            f"{self.export_path}_best_params.json"
        )
        dump(self.bayes_cv_tuner, f"{self.export_path}_bayes_search.pkl")

    def print_status(self, optim_results):
        print(
            f"""
Model #{len(opt.bayes_cv_tuner.cv_results_['params'])}
Best: {self.bayes_cv_tuner.best_score_}
Best params: {self.bayes_cv_tuner.best_params_}
        """
        )


params = {
    "n_jobs": 2,
    "booster": "gbtree",
    "objective": "reg:squarederror",
}


default_params = {
    "n_jobs": 2,
    "booster": "gbtree",
    "objective": "reg:squarederror",
    "learning_rate": 0.01,#Step size shrinkage
    "n_estimators": 1000,#set to 1000 cause we don't have alot of entry data. Is the number of tree
    "max_depth": 3,#stand for maximum depth of a tree, more deep it is, more likey the model will overfit
    "subsample": 0.8,#Subsample ratio of training instance to prevent overfit
    "colsample_bytree": 1,#subsample ratio of columns when constructing each tree
    "gamma": 1 #Minimum loss reduction, the larger it is, the more algo is conservative
}

model1 = xgb.XGBRegressor(**default_params)
eval_metric = ["rmse"]
metric = {}
best_iteration = {}
for i in y_train.columns:
    #eval_set = [(X_train[X_columns_name_Guo[i]], y_train[i]), (X_test[X_columns_name_Guo[i]], y_test[i])]
    eval_set = [(X_train, y_train[i]), (X_test, y_test[i])]
    #model1.fit(X_train[X_columns_name_Guo[i]], y_train[i], eval_metric=eval_metric, eval_set=eval_set, early_stopping_rounds=10, verbose=False)
    model1.fit(X_train, y_train[i], eval_metric=eval_metric, eval_set=eval_set, early_stopping_rounds=10, verbose=False)
    metric[i] = model1.evals_result()
    best_iteration[i] = model1.best_iteration

estimator_min = min(best_iteration.values()) - 100
estimator_max = max(best_iteration.values()) + 100


OPT =  False
ITERATIONS = 100

if OPT:
    for itarget in y_train.columns:
        OPT_EXPORT = path+"/results/template/opt/XGboost_opt_allfeatures/"+itarget
        X = X_train[X_columns_name_Guo[itarget]]

        search_spaces = {
            "learning_rate": (0.001, 1),
            "max_depth": (3, 10),
            "subsample": (0.5, 1),
            "n_estimators": (estimator_min, estimator_max),
        }

        opt = BayesSearch(
            model=xgb.XGBRegressor(*params, **params),
            search_spaces=search_spaces,
            n_iter=ITERATIONS,
            export_path=OPT_EXPORT,
        )

        opt.fit(X, y_train[itarget])

USE_OPT = False

boosted = {}
xgboost_rmse_scores = {}
xgboost_mape_scores = {}

scoring = {'rmse': 'neg_root_mean_squared_error',
           'mape': 'neg_mean_relative_error'
           }


for itarget, itarget_name in enumerate(y_train.columns):
    print(f"\t{itarget}. {itarget_name}\n")

    #X = X_train[X_columns_name_Guo[itarget_name]]
    X = X_train

    opt_params = (
        load(f"./template/opt/XGboost_opt_allfeatures/{itarget_name}_bayes_search.pkl").best_params_ if USE_OPT else {}
    )

    boosted[itarget_name] = xgb.XGBRegressor(
        **{**params, **opt_params}, random_state=RANDOM_SEED
    )
    boosted[itarget_name].fit(
        X,
        y_train[itarget_name],
    )

    scores_XGB = cross_validate(boosted[itarget_name], X, y_train[itarget_name],
                                  scoring=scoring, cv=10)

    scores_XGB['test_rmse'] = -scores_XGB['test_rmse']
    scores_XGB['test_mape'] = np.abs(scores_XGB['test_mape'])

    xgboost_rmse_scores[itarget_name] = scores_XGB['test_rmse']
    xgboost_mape_scores[itarget_name] = scores_XGB['test_mape']

    print(display_scores(xgboost_rmse_scores[itarget_name], xgboost_mape_scores[itarget_name]))
    print(f'{"-" * 30}\n')


# Obtain result on test set

XGB_results, array_results_rmse, array_results_mape = model_result(boosted, X_test, y_test)

# Bayesian Estimation Supersedes the T-Test
# This model replicates the example used in: Kruschke, John. (2012) Bayesian estimation supersedes the t-test. Journal of Experimental Psychology: General.

def best(data, value, group):
    """
    This model replicates the example used in:
    Kruschke, John. (2012) Bayesian estimation supersedes the t-test. Journal of Experimental Psychology: General.
    The original model is extended to handle multiple groups.

    Parameters
    ----------
    data: pandas.DataFrame
        Tidy pandas dataframe
    value: str
        Name of the column holding the values
    group: str
        Name of the column holding the groups
    Returns
    -------
    pymc3.Model
    """
    groups = data[group].unique()

    # pooled empirical mean of the data and twice the pooled empirical standard deviation
    mu = data[value].mean()
    sd = data[value].std() * 2

    # group standard deviations priors
    σ = [0.5, 3]

    with pm.Model() as model:
        groups_means = {
            igroup: pm.Normal(f"{igroup}_mean", mu=mu, sd=sd) for igroup in groups
        }
        groups_std = {
            igroup: pm.Uniform(f"{igroup}_std", lower=σ[0], upper=σ[-1])
            for igroup in groups
        }

        # prior for ν exponentially distributed with a mean of 30
        ν = pm.Exponential("ν_minus_one", lam=1 / 29.0) + 1

        # precision (transformed from standard deviations)
        λ = {igroup: groups_std[igroup] ** -2 for igroup in groups}

        likelihoods = {
            igroup: pm.StudentT(
                igroup,
                nu=ν,
                mu=groups_means[igroup],
                lam=λ[igroup],
                observed=data.query(f'{group} == "{igroup}"')[value].dropna(),
            )
            for igroup in groups
        }

        delta_means, delta_std, effect_size = {}, {}, {}
        for a, b in combinations(groups, 2):
            a_minus_b = f"{a} - {b}"
            delta_means[a_minus_b] = pm.Deterministic(
                f"Δμ ({a_minus_b})", groups_means[a] - groups_means[b]
            )
            delta_std[a_minus_b] = pm.Deterministic(
                f"Δσ ({a_minus_b})", groups_std[a] - groups_std[b]
            )
            effect_size[a_minus_b] = pm.Deterministic(
                f"effect size ({a_minus_b})",
                delta_means[a_minus_b]
                / np.sqrt((groups_std[a] ** 2 + groups_std[b] ** 2) / 2),
            )
    return model


SAMPLE = True
MODEL_PATH = Path("./results/model/Bay_Supersedes")

trace = {}
for itarget, itarget_name in enumerate(y_train.columns):
    print(f"\t{itarget}. {itarget_name}\n")

    d = pd.DataFrame(
        {"test": y_test[itarget_name], "pred": boosted[itarget_name].predict(X_test)}
    ).melt(var_name="group", value_name="mva")

    m = best(data=d, value="mva", group="group")
    with m:
        if SAMPLE:
            trace[itarget_name] = pm.sample(10_000, tune=10_000, random_seed=RANDOM_SEED)
            pm.save_trace(
                trace[itarget_name], directory=MODEL_PATH / itarget_name, overwrite=True
            )
        else:
            trace[itarget_name] = pm.load_trace(MODEL_PATH / itarget_name, model=m)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

fig, ax = plt.subplots(nrows=6, ncols=2, sharex=True, figsize=(12, 8))
for i, (itarget, itrace) in enumerate(trace.items()):
    pm.plot_posterior(
        itrace, varnames=["Δμ (test - pred)"], ref_val=0, round_to=2, ax=ax[i, 0]
    )
    ax[i, 0].set_ylabel(itarget)
    pm.plot_posterior(
        itrace,
        varnames=["effect size (test - pred)"],
        ref_val=0,
        round_to=2,
        ax=ax[i, 1],
    )
ax[i, 0].set_xlabel("Δμ")
ax[i, 1].set_xlabel("d")
plt.tight_layout()
plt.savefig(path+"/results/model/Bay_Supersedes/test.pdf")


for itarget, itrace in trace.items():
    trace_vars = pm.trace_to_dataframe(itrace)
    pm.plot_posterior(itrace, ref_val=0)
    plt.suptitle(itarget)
    plt.tight_layout()

#Save the model
output = open("./model/final/XGboost_noopt.pkl", "wb")
pickle.dump(boosted, output)
output.close()

#Save the model and results
output = open(path+"/results/test_set_result/article/XGB_noopt_array_results_rmse.pkl", "wb")
pickle.dump(array_results_rmse, output)
output.close()

output = open(path+"/results/test_set_result/article/XGB_noopt_array_results_mape.pkl", "wb")
pickle.dump(array_results_mape, output)
output.close()

output = open(path+"/results/test_set_result/article/XGB_noopt.pkl", "wb")
pickle.dump(XGB_results, output)
output.close()

# # Make Df for futher model comparison
# xgboost_rmse_scores = pd.DataFrame.from_dict(xgboost_rmse_scores, orient="columns")
# mean_cv = xgboost_rmse_scores.mean()
# std_cv = xgboost_rmse_scores.std()
# xgboost_rmse_scores = xgboost_rmse_scores.append([mean_cv, std_cv], ignore_index=True)
# xgboost_rmse_scores.rename({xgboost_rmse_scores.index[-2]: 'mean', xgboost_rmse_scores.index[-1]: 'std'}, inplace=True)
#
#
# xgboost_mape_scores = pd.DataFrame.from_dict(xgboost_mape_scores, orient="columns")
# mean_cv = xgboost_mape_scores.mean()
# std_cv = xgboost_mape_scores.std()
# xgboost_mape_scores = xgboost_mape_scores.append([mean_cv, std_cv], ignore_index=True)
# xgboost_mape_scores.rename({xgboost_mape_scores.index[-2]: 'mean', xgboost_mape_scores.index[-1]: 'std'}, inplace=True)


# #Save DF
# cv_score_rmse = xgboost_rmse_scores
# cv_score_rmse.to_pickle(path+"/results/cv/boost_opt/cv15_score_rmse_scale_relativeGB_opt_no_nan.pkl")
#
# cv_score_mape = xgboost_mape_scores
# cv_score_mape.to_pickle(path+"/results/cv/boost_opt/cv15_score_mape_scale_relativeGB_opt_no_nan.pkl")


# # Fit model using each importance as a threshold
# col_name = ['target_name', 'n_features', 'Error %', 'STD', 'Threshold']
# feature_importance = pd.DataFrame(columns=col_name)
# df_best_thres_value = pd.DataFrame()
# mask = {}
# df_X_train_set_prepared = pd.DataFrame(X_train_set_prepared, columns=X_columns_name)
# for target_name in y_train_set:
#     model_feature_importance = boosted[target_name].feature_importances_
#     thresholds = np.sort(boosted[target_name].feature_importances_)
#
#
#     for thresh in thresholds:
#         # select features using threshold
#         selection = SelectFromModel(boosted[target_name], threshold=thresh, prefit=True)
#         select_X_train = selection.transform(X_train_set_prepared)
#         # train model
#         selection_model = XGBRegressor()
#         selection_model.fit(select_X_train, y_train_set[target_name])
#         # eval model
#         pred = cross_validate(selection_model, select_X_train, y_train_set[target_name], scoring='neg_mean_relative_error', cv=10)
#         std = np.abs(pred['test_score'].std())
#         pred = np.abs(pred['test_score'].mean())
#
#         #Append feature_importance_df
#         feature_importance = feature_importance.append({'target_name': target_name,
#                                                         'n_features': select_X_train.shape[1],
#                                                         'Error %': pred,
#                                                         'STD': std,
#                                                         'Threshold': thresh}, ignore_index=True)
#
#     ind_thres = feature_importance[feature_importance.target_name == target_name]
#     best_value = ind_thres[ind_thres['Error %'] == ind_thres['Error %'].min()]
#     df_best_thres_value = df_best_thres_value.append(best_value)
#
#     # Make a list of feature to keep
#     thres = df_best_thres_value.loc[df_best_thres_value['target_name'] == target_name].Threshold.iloc[0]
#     mask_target = model_feature_importance >= thres
#     mask[target_name] = np.array(X_columns_name)[mask_target].tolist()

# # Retake the optimization with only some of the features
# OPT = True
# ITERATIONS = 150
#
# params = {
#     "n_jobs": 2,
#     "booster": "gbtree",
#     "objective": "reg:squarederror"
#     #    "tree_method": "approx",
# #    "objective": "reg:logistic"
# }
#
#
# if OPT:
#     for itarget in y_train_set.columns:
#
#         mask_target = np.where(np.isin(X_columns_name, mask[itarget]))
#
#         OPT_EXPORT = path+"/results/template/opt/opt_median_not_scale_relativeGB_no_nan_feature_selection_10CV/"+itarget
#
#         search_spaces = {
#             "learning_rate": (0.01, 1.0, "log-uniform"),
#             "min_child_weight": (0, 10),
#             "max_depth": (0, 50),
#             "max_delta_step": (0, 40),
#             "subsample": (0.01, 1, "uniform"),
#             "colsample_bytree": (0.01, 1, "uniform"),
#             "colsample_bylevel": (0.01, 1, "uniform"),
#             "reg_lambda": (1e-12, 1000, "log-uniform"),
#             "reg_alpha": (1e-12, 1, "log-uniform"),
#             "gamma": (1e-12, 3.5, "log-uniform"),
#             "min_child_weight": (0, 5),
#             "n_estimators": (20, 120),
#             "scale_pos_weight": (1e-9, 500, "log-uniform"),
#         }
#
#         opt = BayesSearch(
#             model=xgb.XGBRegressor(*params, **params),
#             search_spaces=search_spaces,
#             n_iter=ITERATIONS,
#             export_path=OPT_EXPORT,
#         )
#         opt.fit(X_train_set_prepared[:, mask_target[0]], y_train_set[itarget])
#
# USE_OPT = True
#
# boosted = {}
# xgboost_rmse_scores = {}
# xgboost_mape_scores = {}
#
# scoring = {'nmse': 'neg_mean_squared_error',
#            'mape': 'neg_mean_relative_error'
#            }
#
#
# for itarget, itarget_name in enumerate(y_train_set.columns):
#     print(f"\t{itarget}. {itarget_name}\n")
#
#     opt_params = (
#         load(f"./template/opt/opt_median_not_scale_relativeGB_no_nan_feature_selection_10CV/{itarget_name}_bayes_search.pkl").best_params_ if USE_OPT else {}
#     )
#
#     boosted[itarget_name] = xgb.XGBRegressor(
#         **{**params, **opt_params}, random_state=RANDOM_SEED
#     )
#     boosted[itarget_name].fit(
#         X_train_set_prepared,
#         y_train_set[itarget_name],
#     )
#
#     scores_XGB = cross_validate(boosted[itarget_name], X_train_set_prepared, y_train_set[itarget_name],
#                                   scoring=scoring, cv=10)
#
#     scores_XGB['test_nmse'] = np.sqrt(-scores_XGB['test_nmse'])
#     scores_XGB['test_mape'] = np.abs(scores_XGB['test_mape'])
#
#     xgboost_rmse_scores[itarget_name] = scores_XGB['test_nmse']
#     xgboost_mape_scores[itarget_name] = scores_XGB['test_mape']
#
#     print(display_scores(xgboost_rmse_scores[itarget_name], xgboost_mape_scores[itarget_name]))
#     print(f'{"-" * 30}\n')
