# Load modules
import os
import pandas as pd

os.chdir(".../CBRNet")

# TCGA
path = "exp_tcga_2.csv"

tcga = pd.read_csv(path)

tcga_m = tcga.mean().round(2)
tcga_s = tcga.std().round(2)

tcga = tcga_m.astype(str) + " ± \scriptsize{" + tcga_s.astype(str) + "}"

# IHDP
path = "exp_ihdp_1.csv"

ihdp = pd.read_csv(path)

ihdp_m = ihdp.mean().round(2)
ihdp_s = ihdp.std().round(2)

ihdp = ihdp_m.astype(str) + " ± \scriptsize{" + ihdp_s.astype(str) + "}"

# News
path = "exp_news_2.csv"

news = pd.read_csv(path)

news_m = news.mean().round(2)
news_s = news.std().round(2)

news = news_m.astype(str) + " ± \scriptsize{" + news_s.astype(str) + "}"

# Synth
path = "exp_synth_1.csv"

synth = pd.read_csv(path)

synth_m = synth.mean().round(2)
synth_s = synth.std().round(2)

synth = synth_m.astype(str) + " ± \scriptsize{" + synth_s.astype(str) + "}"

tab = pd.concat([tcga, ihdp, news, synth], axis=1)

tab.columns = ["TCGA-2", "IHDP-1", "News-2", "Synth-1"]

tab.index = [
    "seed",
    "MLP",
    "DRNet",
    "VCNet",
    "CBRNet(MMD$_{lin}$)",
    "CBRNet(MMD$_{rbf}$)",
    "CBRNet(Wass)",
]

print(f"Results for benchmarks")
print(tab.to_latex())
