# Load modules
import os
import pandas as pd

os.chdir(".../CBRNet")

path = "exp_main.csv"

data = pd.read_csv(path)

# Create tables for all levels of bias_intra
biases_intra = data["bias_intra"].unique()

# Create tables for all levels of bias_intra
for bias in biases_intra:
    subset = data[data["bias_intra"] == bias]
    subset = subset.drop(columns=["bias_intra"])
    means = (
        subset.groupby(["bias_inter"])
        .mean()
        .round(2)
        .sort_values("bias_inter", ascending=True)
        .transpose()
        .sort_index()
    )
    stds = (
        subset.groupby(["bias_inter"])
        .std()
        .round(2)
        .sort_values("bias_inter", ascending=True)
        .transpose()
        .sort_index()
    )
    
    strs = means.astype(str) + " ± \scriptsize{" + stds.astype(str) +"}"
    
    strs.index = ['CART', 'CBRNet(MMD$_{lin}$)', 'CBRNet(MMD$_{rbf}$)', 'CBRNet(Wass)',
       'DRNet', 'Linear Regression', 'MLP', 'VCNet', 'xgboost',
       'seed']
    
    strs = strs.reindex(['Linear Regression', 'CART', 'xgboost', 'MLP', 'DRNet', 'VCNet', 'CBRNet(MMD$_{lin}$)', 'CBRNet(MMD$_{rbf}$)', 'CBRNet(Wass)',
       'seed'])
    
    print(f"Results for bias_intra = {bias}")
    print(strs.to_latex())

# Create tables for all levels of bias_inter
biases_inter = data["bias_inter"].unique()

# Create tables for all levels of bias_intra
for bias in biases_inter:
    subset = data[data["bias_inter"] == bias]
    subset = subset.drop(columns=["bias_inter"])
    means = (
        subset.groupby(["bias_intra"])
        .mean()
        .round(2)
        .sort_values("bias_intra", ascending=True)
        .transpose()
        .sort_index()
    )
    stds = (
        subset.groupby(["bias_intra"])
        .std()
        .round(2)
        .sort_values("bias_intra", ascending=True)
        .transpose()
        .sort_index()
    )
    
    strs = means.astype(str) + " ± \scriptsize{" + stds.astype(str) +"}"
    
    strs.index = ['CART', 'CBRNet(MMD$_{lin}$)', 'CBRNet(MMD$_{rbf}$)', 'CBRNet(Wass)',
       'DRNet', 'Linear Regression', 'MLP', 'VCNet', 'xgboost',
       'seed']
    
    strs = strs.reindex(['Linear Regression', 'CART', 'xgboost', 'MLP', 'DRNet', 'VCNet', 'CBRNet(MMD$_{lin}$)', 'CBRNet(MMD$_{rbf}$)', 'CBRNet(Wass)',
       'seed'])
    
    print(f"Results for bias_inter = {bias}")
    print(strs.to_latex())
