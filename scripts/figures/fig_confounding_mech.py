# Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scienceplots
plt.style.use('science')
from scipy.stats import beta

# Set ws
import os
os.chdir(".../CBRNet")


######################
# Generate plots
######################

# Get dosages to plot

dosages = np.linspace(np.finfo(float).eps, 1-np.finfo(float).eps, 128)

# Get number of clusters
num_clusters = 3

for i, j in zip([0,3,3,3],[0, 0, 0.33, 0.67]):
    # Get cluster modes
    lb = 0.5 - 0.5 * j
    ub = 0.5 + 0.5 * j
    dosage_modes_per_cluster = np.linspace(lb,ub,num_clusters)

    a = 1 + i

    ## Cluster 1
    # Calculate beta
    b = (a - 1.0) / dosage_modes_per_cluster[0] + (2.0 - a)

    # Get pdf
    pdf1 = [beta.pdf(d,a,b) for d in dosages]

    ## Cluster 2
    # Calculate beta
    b = (a - 1.0) / dosage_modes_per_cluster[1] + (2.0 - a)

    # Get pdf
    pdf2 = [beta.pdf(d,a,b) + 0.01 for d in dosages]

    ## Cluster 3
    # Calculate beta
    b = (a - 1.0) / dosage_modes_per_cluster[2] + (2.0 - a)

    # Get pdf
    pdf3 = [beta.pdf(d,a,b) + 0.02 for d in dosages]

    ## Plot
    fig = plt.figure(figsize=(4,4))

    plt.ylim(0,6)
    plt.xlabel('Dosage', fontsize=20)
    plt.ylabel('Prob.', fontsize=20) 

    plt.plot(dosages, pdf1, color=cm.viridis(0.1))
    plt.plot(dosages, pdf2, color=cm.viridis(0.5))
    plt.plot(dosages, pdf3, color=cm.viridis(0.9))
    
    fig.set_size_inches(4,4)
    
    plt.savefig(f"fig_confounding_mech_{i}_{j}.pdf")