## KEY SETTINGS
#####################################

w = 4
h = 3

# Load modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import MaxNLocator
plt.style.use('science')

os.chdir(".../CBRNet")

path = "res/exp_cluster_impact.csv"
data = pd.read_csv(path)

# Rm mise prefix
data.columns = [col.replace("MISE ", "") for col in data.columns]

# Scale with resp to cluster = 1 (no reg)
for s in data.seed.unique():
    data.loc[data.seed == s,("cbrnet-rbf", "cbrnet-lin", "cbrnet-was")] = \
        data.loc[data.seed == s,("cbrnet-rbf", "cbrnet-lin", "cbrnet-was")] / \
            data.loc[data.seed == s,("cbrnet-rbf", "cbrnet-lin", "cbrnet-was")].iloc[0,:]

means = (
    data.groupby(["cluster"])
    .mean()
    .round(2)
    .sort_values("cluster", ascending=True)
    .transpose()
    .sort_index()
)

means.drop(["seed"], inplace=True)

stds = (
    data.groupby(["cluster"])
    .std()
    .round(2)
    .sort_values("cluster", ascending=True)
    .transpose()
    .sort_index()
)

stds.drop(["seed"], inplace=True)

# Transpose the DataFrame so that rows become columns and vice versa
transposed_means = means.transpose()
transposed_stds = stds.transpose()

# Create a line plot
fig = plt.figure()

ax0 = fig.add_axes([0, 0, 1, 1])
ax0.axis('off')

ax1 = fig.add_axes([0.15,0.2,0.9,0.9])

# Set lower lim
#ax1.set_ylim(bottom=0)

markers = ['x','x','x']

for column,marker in zip(transposed_means.columns, markers):
    ax1.plot(transposed_means.index, transposed_means[column], marker=marker, label=column)
    ax1.fill_between(transposed_means.index, transposed_means[column] - transposed_stds[column], transposed_means[column] + transposed_stds[column], alpha=0.2)

ax1.set_ylabel('MISE', fontsize=20)
ax1.set_xlabel('Number of Clusters', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# ax1.set_ylim(bottom=0)
# ax1.set_ylim(top=1.2)

fig.set_size_inches(w,h)

plt.savefig("cluster_impact.pdf")

fig_leg = plt.figure(figsize=(1, 2))
ax_leg = fig_leg.add_subplot(111)

new_labels = ['CBRNet+MMD(lin)', 'CBRNet+MMD(rbf)', 'CBRNet+Wass']

# Get the handles from the previous plot
handles, _ = ax1.get_legend_handles_labels()


# Add the legend from the previous plot to the new figure
ax_leg.legend(handles, new_labels)

# Hide the axes of the new figure
ax_leg.axis('off')

fig_leg.savefig("cluster_impact_legend.pdf")

# Generate table
strs = means.astype(str) + " ± \scriptsize{" + stds.astype(str) +"}"
print(strs.to_latex())