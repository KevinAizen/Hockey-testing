# 4) b) Make correlation with is p value
import os
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.chdir("./results")

test = pd.read_pickle("./template/all_data/test_relative_GB.pkl")
test.drop(columns=['Position'], inplace=True)

X_descriptive = pd.read_pickle("./template/descriptive_stats/X_descriptive_relative_GB.pkl")
test.rename(
    columns={"Wingate": "Wattbike"},
    inplace=True
)
features = test.iloc[:, 0:18].columns.tolist()

target = test.iloc[:, 18:].columns.tolist()

test.dropna(inplace=True)
test.reset_index(drop=True, inplace=True)

corr_matrix = test.corr()

corr_matrix = corr_matrix.loc[target, features].T

n = 72
dist = ss.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
p = 2*dist.cdf(-abs(corr_matrix))

p = pd.DataFrame(p, columns=target, index=features)


labels = corr_matrix.round(2).astype(str)
p_value = p
for i in labels:
    print(i)
    for index, value in labels[i].items():
        print(index, value)
        if p_value.loc[index, i] <= 0.01:
            labels.loc[index, i] = value + '**'
        elif p_value.loc[index, i] <= 0.05:
            labels.loc[index, i] = value + '*'
        else:
            labels.loc[index, i] = value

labels = np.asarray(labels)

# Heatmap with Sprint time (s)
cmap = sns.diverging_palette(233, 233, as_cmap=True)
fig_hmap = plt.figure(figsize=(11, 9))
fig_hmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=labels, fmt='', linewidths=0.5, cmap=cmap, annot_kws={"size": 14})
cbar = fig_hmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)


plt.xlabel('On-ice tests', fontsize=20)
plt.xticks(fontsize=17)
plt.ylabel('Off-ice tests', fontsize=20)
plt.yticks(fontsize=17)
#plt.title('Pearson correlation between off-ice and on-ice tests', fontsize=20, pad=20)
plt.tight_layout()
# save fig, df
plt.savefig('./pearsonr.png', dpi=500)
#
# with pd.ExcelWriter('./template/descriptive_stats/stats_pearson.xlsx') as writer:
#     corr_matrix.to_excel(writer, sheet_name='Corrélation Pearson')
#     p.to_excel(writer, sheet_name='p_value')