from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import shap
from IPython.display import display, HTML

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.chdir('/Users/kevinaizen/PycharmProjects/analysis_groinbar/machine_learning/')

NORMALIZE = 'False'
X = pd.read_pickle('X.pkl')
X.drop(
    columns='Position',
    inplace=True
)
y = pd.read_pickle('y.pkl')

#  Splitting into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

#  Helper functions
def mape(y_test, y_pred):
    return (np.abs((y_test - y_pred) / y_test)) * 100


def mae(y_test, y_pred):
    return np.abs(y_test - y_pred)


def evaluate(y_test, y_pred):
    d = {"mae": mae(y_test, y_pred), "mape": mape(y_test, y_pred)}
    print(
        f"""
    \tmae = {d['mae'].mean():.3f} ({d['mae'].std():.3f})
    \tmape = {d['mape'].mean():.3f}% ({d['mape'].std():.3f})
    """
    )
    return d

# Fit model
OPT = False
ITERATIONS = 100

params = {"n_jobs": 1, "silent": 1, "tree_method": "approx"}

if OPT:
    for itarget in y.columns:
        OPT_EXPORT = f"opt/{itarget}"

        search_spaces = {
            "learning_rate": (0.01, 1.0, "log-uniform"),
            "min_child_weight": (0, 10),
            "max_depth": (0, 50),
            "max_delta_step": (0, 20),
            "subsample": (0.01, 1.0, "uniform"),
            "colsample_bytree": (0.01, 1.0, "uniform"),
            "colsample_bylevel": (0.01, 1.0, "uniform"),
            "reg_lambda": (1e-9, 1000, "log-uniform"),
            "reg_alpha": (1e-9, 1.0, "log-uniform"),
            "gamma": (1e-9, 0.5, "log-uniform"),
            "min_child_weight": (0, 5),
            "n_estimators": (50, 100),
            "scale_pos_weight": (1e-6, 500, "log-uniform"),
        }

        opt = BayesSearch(
            model=XGBRegressor(**params),
            search_spaces=search_spaces,
            n_iter=ITERATIONS,
            export_path=OPT_EXPORT,
        )
        opt.fit(X_train, y_train[itarget])

USE_OPT = False

boosted = {}
evaluation = {}

RANDOM_SEED = 15
np.random.seed(RANDOM_SEED)

for itarget, itarget_name in enumerate(y_train.columns):
    print(f"\t{itarget}. {itarget_name}\n")

    opt_params = (
        load(f"opt/{itarget_name}_bayes_search.pkl").best_params_ if USE_OPT else {}
    )

    boosted[itarget_name] = XGBRegressor(
        **{**params, **opt_params}, random_state=RANDOM_SEED
    )
    boosted[itarget_name].fit(
        X_train,
        y_train[itarget_name],
        eval_set=[(X_train, y_train[itarget_name]), (X_test, y_test[itarget_name])],
        early_stopping_rounds=50,
        eval_metric="rmse",
        verbose=50,
    )

    evaluation[itarget_name] = evaluate(
        y_test[itarget_name], boosted[itarget_name].predict(X_test)
    )
    evaluation[itarget_name]["test"] = [
        itarget_name for i in range(len(evaluation[itarget_name]["mae"]))
    ]

    print(f'{"-" * 30}\n')
evaluation = pd.concat([pd.DataFrame(evaluation[itest]) for itest in evaluation.keys()])


fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(12, 4))
sns.boxplot(y="test", x="mae", data=evaluation, ax=ax[0], orient="h", color="grey")
sns.boxplot(y="test", x="mape", data=evaluation, ax=ax[1], orient="h", color="grey")
ax[0].set_ylabel("")
ax[1].set_ylabel("")
ax[1].set_xlabel("mape (%)")
# TODO: %

ax[0].set_yticklabels(
    [
        "25B\n(sec)",
        "5B\n(sec)",
        "30B\n(sec)",
        "25F\n(sec)",
        "5F\n(sec)",
        "30F\n(sec)",
    ]
)
plt.tight_layout()
sns.despine()
plt.savefig('eval.png', dpi=300)

evaluation.groupby('test').mean()['mape'].sum()

# Interpretation
shap.initjs()

explainer = {itarget: shap.TreeExplainer(boosted[itarget]) for itarget in y.columns}
shap_values = {
    itarget: explainer[itarget].shap_values(X) for itarget in y.columns
}

# How important are our features?
# To get an overview of which features are most important for a model we can plot the SHAP values of every feature for
# every sample.
# The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the
# distribution of the impacts each feature has on the model output.

shap_df = pd.concat(
    [
        pd.DataFrame(shap_values[i], columns=X_train.columns).assign(target=i)
        for i in y.columns
    ]
)

shap_df_abs = shap_df.copy()
shap_df_abs[X_train.columns] = shap_df_abs[X_train.columns].abs()

shap_df_abs = shap_df.copy()
shap_df_abs[X_train.columns] = shap_df_abs[X_train.columns].abs()
sns.catplot(col="target", data=shap_df_abs, color="grey", kind="bar", orient="h")

sns.despine()

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(21, 8))
MAX_DISPLAY = 6

for i, itarget in enumerate(y.columns):
    if i <= 2:
        nrow = 0
        ncol = 0
    else:
        nrow = 1
        ncol = 3

    color_bar = True if i == 2 or i == 5 else False

    plt.sca(ax[nrow, i-ncol])

    shap.summary_plot(
        shap_values[itarget],
        X,
        show=False,
        color_bar=color_bar,
        auto_size_plot=False,
        max_display=MAX_DISPLAY,
        plot_type="dot",
    )
    ax[nrow, i-ncol].set_title(itarget)
    ax[nrow, i-ncol].set_xlim(-0.5, 0.5)
    if i != 4:
        ax[nrow, i-ncol].set_xlabel("")

plt.tight_layout()
sns.despine()
plt.savefig('shap.png', dpi=300)

# What is the prediction path?
# The below explanation shows features each contributing to push the model output from the base value
# (the average model output over the training dataset we passed) to the model output.
# Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.

random_observation = np.random.randint(0, X.shape[0])

itarget = y.columns[0]

shap.force_plot(explainer[itarget].expected_value,
                shap_values[itarget][random_observation, :],
                X.iloc[random_observation, :],
                matplotlib=True)

shap.force_plot(explainer[itarget].expected_value,
                shap_values[itarget],
                X,
                matplotlib=True)
