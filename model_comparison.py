import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from function import get_data, make_comparison_graph, get_dict_data, str_columns_changer, feature_selection_graph, stat_test_model, stat_test_target


path = os.getcwd()
model_path = path+'/results/model/final/'
X_path = path+'/results/template/X_train_set/'
columns_path = path+'/results/template/X_columns_name/'

# Get the result df for testing set
rmse, mape, mean = get_dict_data(path+'/results/test_set_result/article/')
results = rmse.append(mape)

stat_test_model(rmse)
stat_test_target(mape)

dict = {'Lasso_op': 'LCV', 'XGB_noop': 'XGB', 'Lin_reg_': 'LR'}
mean = str_columns_changer(mean, dict_name=dict)
mape = str_columns_changer(mape, dict_name=dict)
rmse = str_columns_changer(rmse, dict_name=dict)
mape['Error_type'], rmse['Error_type'] = 'Mape', 'Rmse'
results = mape.append(rmse, ignore_index=True)

graph_list_order = ['LCV', 'LR', 'XGB']
graph = make_comparison_graph(results, hue_order=graph_list_order)



# Get the feature importance from model
feature_selection_graph(model_path, X_path, columns_path)

# # load x_train_prepared value
# rmse_boost = get_data(path+"/results/cv/boost/", "cv_score_rmse_scale_relativeGB_no_opt_no_nan.pkl", 'cv_boost')
# mape_boost = get_data(path+"/results/cv/boost/", "cv_score_mape_scale_relativeGB_no_opt_no_nan.pkl", 'cv_boost')
#
# # rmse_boost_opt3CV = get_data(path+"/results/cv/boost_opt/", "cv3_score_rmse_scale_relativeGB_opt_no_nan.pkl", 'cv3_boost_opt')0.068
# # mape_boost_opt3CV = get_data(path+"/results/cv/boost_opt/", "cv3_score_mape_scale_relativeGB_opt_no_nan.pkl", 'cv3_boost_opt')
# #
# # rmse_boost_opt5CV = get_data(path+"/results/cv/boost_opt/", "cv5_score_rmse_scale_relativeGB_opt_no_nan.pkl", 'cv5_boost_opt')
# # mape_boost_opt5CV = get_data(path+"/results/cv/boost_opt/", "cv5_score_mape_scale_relativeGB_opt_no_nan.pkl", 'cv5_boost_opt')
#
# rmse_boost_opt10CV = get_data(path+"/results/cv/boost_opt/", "cv10_score_rmse_scale_relativeGB_opt_no_nan.pkl", 'cv10_boost_opt')
# mape_boost_opt10CV = get_data(path+"/results/cv/boost_opt/", "cv10_score_mape_scale_relativeGB_opt_no_nan.pkl", 'cv10_boost_opt')
#
# # rmse_boost_opt15CV = get_data(path+"/results/cv/boost_opt/", "cv15_score_rmse_scale_relativeGB_opt_no_nan.pkl", 'cv15_boost_opt')
# # mape_boost_opt15CV = get_data(path+"/results/cv/boost_opt/", "cv15_score_mape_scale_relativeGB_opt_no_nan.pkl", 'cv15_boost_opt')
#
# rmse_lin_reg = get_data(path+"/results/cv/lin_reg/", "cv_score_rmse_scale_relativeGB_no_nan.pkl", 'cv_lin_reg')
# mape_lin_reg = get_data(path+"/results/cv/lin_reg/", "cv_score_mape_scale_relativeGB_no_nan.pkl", 'cv_lin_reg')
#
# rmse_lasso_reg = get_data(path+"/results/cv/lasso_reg/", "cv_score_rmse_scale_relativeGB_no_nan.pkl", 'cv_lasso_reg')
# mape_lasso_reg = get_data(path+"/results/cv/lasso_reg/", "cv_score_mape_scale_relativeGB_no_nan.pkl", 'cv_lasso_reg')
#
# rmse = rmse_lin_reg.append([rmse_lasso_reg, rmse_boost, rmse_boost_opt10CV], ignore_index=True)
# rmse = rmse.rename(
#     columns={
#         "variable": "Sprint distance (meter)",
#         "value": "Rmse error (sec)"
#     }
# )
#
# mape = mape_lin_reg.append([mape_lasso_reg, mape_boost, mape_boost_opt10CV], ignore_index=True)
# mape = mape.rename(
#     columns={
#         "variable": "Sprint distance (meter)",
#         "value": "mape error (%)"
#     }
# )
#
# figsize = (12, 8)
# fig1 = plt.figure(figsize=figsize)
# make_comparison_CVgraph(rmse)
#
# fig2 = plt.figure(figsize=figsize)
# make_comparison_CVgraph(mape)

# os.chdir(path+"/results/cv/fig/")
# fig1.savefig("cv_all_cv10_rmse_scale_relativeGB_no_nan.png", dpi=500)
# fig2.savefig("cv_all_cv10_mape_scale_relativeGB_no_nan.png", dpi=500)






