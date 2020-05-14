import os
import glob
import pickle
import numpy as np
from numpy import loadtxt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import neg_mean_relative_error, mean_squared_error, explained_variance_score
import shap
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, neg_mean_relative_error, mean_absolute_error

# Define a function to get the rmse score + SD from each target, and Mean abosulte pourcentage error (Mape)
def display_scores(scores1, scores2):
    print("rmse scores", scores1)
    print("rmse mean", scores1.mean())
    print("Standard deviation", scores1.std())
    print("mape mean", scores2.mean())
    print("Standard deviation", scores2.std())

def get_data(path, df, model_name):
    os.chdir(path)
    df = pd.read_pickle(df)
    df = df.iloc[:-2, :]
    df.reset_index(inplace=True, drop=True)
    df['model'] = model_name
    df = df.melt(id_vars=['model'])
    return df

def get_dict_data(path):
    os.chdir(path)
    result = glob.glob(path + '/*.{}'.format('pkl'))
    df_rmse = pd.DataFrame()
    df_mape = pd.DataFrame()
    df_results = pd.DataFrame()
    for i in result:
        if 'array_results' in i:
            if 'rmse' in i:
                dict = pd.read_pickle(i)
                df_dict = pd.DataFrame(dict)
                df_dict = pd.melt(df_dict.reset_index(), id_vars='index')
                df_dict['model'] = i[84:92]
                df_rmse = df_dict.append(df_rmse)
            else:
                dict = pd.read_pickle(i)
                df_dict = pd.DataFrame(dict)
                df_dict = pd.melt(df_dict.reset_index(), id_vars='index')
                df_dict['model'] = i[84:92]
                df_mape = df_dict.append(df_mape)
        else:
            dict = pd.read_pickle(i)
            df_dict = pd.DataFrame(dict)
            df_dict = pd.melt(df_dict, id_vars='target', value_vars=['rmse', 'std_rmse', 'mape', 'std_mape'])
            df_dict['model'] = i[84:92]
            df_results = df_dict.append(df_results)

    return df_rmse, df_mape, df_results

def feature_selection_graph(model_path, X_path, columns_path):
    os.chdir(model_path)
    model = glob.glob(model_path + '/*.{}'.format('pkl'))
    os.chdir(columns_path)
    pkl_file = open('X_columns_name.pkl', 'rb')
    X_columns = pickle.load(pkl_file)
    pkl_file.close()
    os.chdir(X_path)
    for model_name in model:
        os.chdir(X_path)
        X = loadtxt("./X_train_set_prepared_not_scale_relative_GB_no_nan.csv", delimiter=',')
        X = pd.DataFrame(X, columns=X_columns)
        model_dict = pd.read_pickle(model_name)
        if 'XGboost_opt' in model_name:
            os.chdir(columns_path)
            pkl_file = open('X_columns_name_Guo.pkl', 'rb')
            columns = pickle.load(pkl_file)
            pkl_file.close()

            sep = ' '
            model_name = model_name.split('/')[8].split('_')
            model_name = [model_name[0], model_name[-1].split('.')[0]]
            model_name = sep.join(model_name)

            shap_graph(X, column_names=columns, model=model_dict, max_display=5, model_name=model_name)

        elif 'XGboost_noopt' in model_name:

            sep = ' '
            model_name = model_name.split('/')[8].split('_')
            model_name = [model_name[0], model_name[-1].split('.')[0]]
            model_name = sep.join(model_name)
            model_name = 'XGB'

            shap_graph(X, column_names=X_columns, model=model_dict, max_display=5, model_name=model_name)

        else:
            model_dict = pd.read_pickle(model_name)

            sep = ' '
            model_name = model_name.split('/')[8].split('_')
            model_name = [model_name[0], model_name[1]]
            model_name = sep.join(model_name)
            model_name = 'LCV'

            coef_graph(X_columns, model_dict, max_display=5, model_name=model_name)


def shap_graph(X, column_names, model, max_display=5, model_name=None):
    summary_plot, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(11, 9))
    #summary_plot.subplots_adjust(hspace=0.4, wspace=1)
    summary_plot.text(0.01, 0.5, 'Features', va='center', rotation='vertical', fontsize=20)
    summary_plot.text(0.5, 0.01, 'SHAP value (impact on the model output)', ha='center', rotation='horizontal', fontsize=20)
    #summary_plot.suptitle('Features Importance Shap values of '+ model_name , fontsize=22)
    column_names[5] = 'Wattbike'

    for target in model:
        if 'XGboost opt' in model_name:
            X_df = X[column_names[target]]
            feature_names = column_names[target]
        else:
            X_df = X
            feature_names = column_names
        ax = list(model.keys()).index(target)
        explainer = shap.TreeExplainer(model[target])
        shap_values = explainer.shap_values(X_df)
        if 'F' in target:
            axy = 0
            axx = ax
        else:
            axy = 1
            axx = ax - 3
        color_bar = True if axx == 2 else False
        plt.sca(axs[axy, axx])
        shap.summary_plot(
                          shap_values,
                          X_df,
                          plot_type="violin",
                          feature_names=feature_names,
                          plot_size=None,
                          color_bar=color_bar,
                          max_display=max_display
                          )
        axs[axy, axx].set_title(target, fontsize=18)
        if axx == 0:
            axs[axy, axx].set_xlim(-0.10, 0.10)
        else:
            axs[axy, axx].set_xlim(-0.30, 0.30)
        #if ax != 4:
        axs[axy, axx].set_xlabel("")

        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    return summary_plot

def coef_graph(column_names, model, max_display=5, model_name=None):
    sns.set(style="whitegrid")
    summary_plot, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(11, 9))
    #summary_plot.subplots_adjust(hspace=0.2, wspace=0.6)
    summary_plot.text(0.01, 0.5, 'Features', va='center', rotation='vertical', fontsize=20)
    summary_plot.text(0.5, 0.01, 'Regression coefficient of feature on model output', ha='center', rotation='horizontal', fontsize=20)
    #summary_plot.suptitle('Features Importance of '+ model_name, fontsize=24)
    column_names[5] = 'Wattbike'

    for target in model:
        coef = model[target].coef_
        coef_abs = np.abs(model[target].coef_)
        df = pd.DataFrame({'variable': column_names, 'value': coef, 'value_abs': coef_abs}, columns=['variable', 'value', 'value_abs'])
        zero_coef = (df['value'] == 0).astype(int).sum(axis=0)
        if zero_coef <= 13:
            max_feature = max_display
        else:
            max_feature = int(df['value'].astype(bool).sum(axis=0))
        df = df.sort_values('value_abs', ascending=False)
        df = df.reset_index(drop=True).iloc[:max_feature, :]
        ax = list(model.keys()).index(target)
        if 'F' in target:
            axy = 0
            axx = ax
        else:
            axy = 1
            axx = ax - 3

        sns.barplot(
            x='value',
            y='variable',
            data=df,
            color='b',
            ax=axs[axy, axx],
        )

        # axs[axy, axx].barh(df.variable, df.value)

        # height = 0.8
        # for patch in axs[axy, axx].patches:
        #     current_height = patch.get_height()
        #     if current_height != height:
        #         diff = current_height - height
        #
        #         # we change the bar height
        #         patch.set_height(height)
        #
        #         # we recenter the bar
        #         patch.set_y(patch.get_y() + diff * 0.5)

        axs[axy, axx].set_ylim(max_display-0.5, -0.5)
        axs[axy, axx].set_title(target, fontsize=18)
        axs[axy, axx].tick_params(axis='y', labelsize=17)
        axs[axy, axx].tick_params(axis='x', labelsize=17)


        for p in axs[axy, axx].patches:
            if p.get_width() < 0:
                _x = -p.get_x()
                _y = p.get_y() + p.get_height() - (p.get_height() / 2)
                value = format(p.get_width(), ".4f")
                axs[axy, axx].text(_x, _y, value, va='center', fontsize=14, fontweight='bold')
            else:
                _x = p.get_x()
                _y = p.get_y() + p.get_height() - (p.get_height() / 2)
                value = format(p.get_width(), ".4f")
                axs[axy, axx].text(_x, _y, value, va='center', ha='right', fontsize=14, fontweight='bold')

        if ax == 5:
            axs[axy, axx].set_xlim(-0.07, 0.07)
        else:
            axs[axy, axx].set_xlim(-0.05, 0.05)
        axs[axy, axx].set_ylabel("")
        #if ax != 4:
        axs[axy, axx].set_xlabel("")
        # else:
        #     axs[axy, axx].set_xlabel("Regression coefficient of feature on model output", fontsize=20)


    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    return summary_plot

def make_comparison_graph(df, hue_order=None):
    sns.set(style="whitegrid")
    comparaison, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(14, 8))
    comparaison.subplots_adjust(hspace=0.4, wspace=1)
    comparaison.suptitle('Models prediction error on testing set', fontsize=20)

    if hue_order:
        hue_order = hue_order

    for error in df.Error_type.unique():
        if error == 'Rmse':
            ax = 0
        elif error == 'Mape':
            ax = 1

        new_df = df['Error_type'] == error
        new_df = df[new_df]
        sns.boxplot(
                        x=new_df.columns[1],
                        y=new_df.columns[2],
                        hue=new_df.columns[3],
                        hue_order=hue_order,
                        data=new_df,
                        palette="Blues",
                        ax=axs[ax]
                    )
        if error == 'Rmse':
            axs[ax].set_ylabel('Root mean squared error (sec)', fontsize=18)
            axs[ax].set_xlabel("(A)", fontsize=16)
            axs[ax].set_ylim(0, 0.8)
            axs[ax].tick_params(axis='x', labelsize=16)
            axs[ax].tick_params(axis='y', labelsize=16)
            handles, _ = axs[ax].get_legend_handles_labels()
            axs[ax].legend(handles, ["LCV", "LR", "XGB"], loc='upper left')

        elif error == 'Mape':
            axs[ax].set_ylabel('Mean absolute percentage error (%)', fontsize=18)
            axs[ax].set_xlabel("(B)", fontsize=16)
            axs[ax].set_ylim(0, 26)
            axs[ax].tick_params(axis='x', labelsize=16)
            axs[ax].tick_params(axis='y', labelsize=16)
            axs[ax].get_legend().remove()


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])



def str_columns_changer(df, columns='model', dict_name=None):
    df[columns] = df[columns].replace(dict_name)
    return df


def Confidence_interval_95(describe_df):
    '''compute confidence interval 5 - 95 %'''
    describe_index_name = list(describe_df.index)
    describe_index_name = describe_index_name + ["Ci_95", "Ci_5"]

    ci95_hi = []
    ci95_lo = []

    ci95_hi = describe_df.loc["mean", :] + 1.96 * describe_df.loc["std", :] / np.sqrt(describe_df.loc["count", :])
    ci95_lo = describe_df.loc["mean", :] - 1.96 * describe_df.loc["std", :] / np.sqrt(describe_df.loc["count", :])

    describe_df = describe_df.append([ci95_hi, ci95_lo], ignore_index=True)
    describe_df.index = describe_index_name

    return describe_df

#def barplot_feature_coef(x, y):

def model_result(model, X_test, y_true, columns=None):
    result = pd.DataFrame(columns=['target', 'rmse','std_rmse', 'mape', 'std_mape'])
    array_result_rmse = {}
    array_result_mape = {}
    for target in model:
        if columns:
            X = X_test[columns[target]]
            y_pred = model[target].predict(X)
        else:
            y_pred = model[target].predict(X_test)
        y = np.array(y_true[target])
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        std_rmse = np.sqrt(np.std((y - y_pred) ** 2))
        std_mape = np.std(np.abs((y - y_pred) / y)) * 100
        r_squared = r2_score(y, y_pred)
        result = result.append({'target': target, 'rmse': rmse, 'std_rmse':std_rmse, 'mape': mape, 'std_mape':std_mape, 'r_squared': r_squared}, ignore_index=True)
        array_result_rmse[target] = np.sqrt((y - y_pred)**2)
        array_result_mape[target] = np.abs(((y - y_pred) / y)) * 100
    return result, array_result_rmse, array_result_mape

def stat_test_model(df):
    for target in df.variable.unique():
        new_df = df['variable'] == target
        new_df = df[new_df]
        f, p = stats.f_oneway(new_df['value'][new_df['model'] == 'XGB_noop'],
                              new_df['value'][new_df['model'] == 'Lasso_op'],
                              new_df['value'][new_df['model'] == 'Lin_reg_'])

        print('One-way ANOVA : ' + target)
        print('=============')
        print('F value:', f)
        print('P value:', p, '\n')
        print('=============')

        if p <= 0.05:
            mc = MultiComparison(new_df['value'], new_df['model'])
            result = mc.tukeyhsd()
            print(result)
            print('=============')

def stat_test_target(df):
    for model in df.model.unique():
        new_df = df['model'] == model
        new_df = df[new_df]
        f, p = stats.f_oneway(new_df['value'][new_df['variable'] == '5F'],
                              new_df['value'][new_df['variable'] == '5_30F'],
                              new_df['value'][new_df['variable'] == '30F'],
                              new_df['value'][new_df['variable'] == '5B'],
                              new_df['value'][new_df['variable'] == '5_30B'],
                              new_df['value'][new_df['variable'] == '30B'])

        print('One-way ANOVA : ' + model)
        print('=============')
        print('F value:', f)
        print('P value:', p, '\n')
        print('=============')

        if p <= 0.05:
            mc = MultiComparison(new_df['value'], new_df['variable'])
            result = mc.tukeyhsd()
            print(result)
            print('=============')


def backwardElimination(X_train, y_train, X_columns, sl):
    X_train = pd.DataFrame(X_train, columns=X_columns)
    y_train.reset_index(drop=True, inplace=True)
    X = X_train

    # backwardElimination
    if len(X.columns) > 1:
        vars = len(X.columns)
        for i in range(0, vars):
            X2 = sm.add_constant(X)
            est = sm.OLS(y_train, X2)
            est2 = est.fit()
            maxVar = max(est2.pvalues)
            if maxVar > sl:
                for j in range(1, vars + 1 - i):
                    if est2.pvalues[j] == maxVar:
                        X = X.drop(est2.pvalues.index[j], axis=1)
        print(est2.summary())

    else:
        X2 = sm.add_constant(X)
        est = sm.OLS(y_train, X2)
        est2 = est.fit()
        print(est2.summary())

    return X

