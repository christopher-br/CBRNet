# LOAD MODULES
# Standard library
from typing import Callable

# Third party
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
from matplotlib import colormaps as cm

# CUSTOM FUNCTIONS
def normalize_variables(x: np.ndarray) -> np.ndarray:
    """
    Normalizes the variables in the given covariates matrix.

    The function normalizes each variable (column) in the covariates matrix to the range [0, 1] using min-max normalization.

    Parameters:
    x (np.ndarray): The covariates matrix, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.

    Returns:
    np.ndarray: The normalized covariates matrix.
    """
    num_covariates = x.shape[1]
    for idx in range(num_covariates):
        x[:, idx] = (x[:, idx] - np.min(x[:, idx])) / (np.max(x[:, idx]) - np.min(x[:, idx]))
    
    return x
        
def tsne_plot(
    x: np.ndarray,
    d: np.ndarray,
    w: float = 4,
    h: float = 4,
    file_name: str = "tsne_plot.pdf",
    labels: bool = True,
):
    col_map = "viridis"
    
    tsne = TSNE(n_components=2,verbose=0,perplexity=30,n_iter=1000)
    tsne_results = tsne.fit_transform(x)
    
    fig, axes = plt.subplot_mosaic("A")
    fig.set_size_inches(w,h)
    
    axes["A"].scatter(
            x=tsne_results[:,0],
            y=tsne_results[:,1],
            alpha=1,
            c=cm[col_map](d),
            s=15,
            edgecolors='none',
        )
    
    if labels is False:
        # Remove the y-axis
        axes["A"].set_yticks([])
        axes["A"].set_xticks([])
    else:
        axes["A"].set_xlabel("Dimension 1", fontsize=20)
        axes["A"].set_ylabel("Dimension 2", fontsize=20)
    
    
    
    plt.savefig(file_name)
    
def dose_plot(
    d: np.ndarray,
    w: float = 4,
    h: float = 1,
    samples: int = None,
    file_name: str = "dose_plot.pdf",
    color: str = None,
    num_bins: int = 33,
    labels: bool = True,
):
    col_map = "viridis"
    
    # Set number of samples
    if samples is None:
        samples = len(d)
    else:
        samples = min(samples, len(d))
        
    # Sample
    ids = np.random.choice(len(d), samples, replace=False)
    d = d[ids]
    
    # Get histogram
    hist, hist_edges = np.histogram(d, num_bins, density=True, range=(0,1))
    
    fig, axes = plt.subplot_mosaic("A")
    
    if color is None:
        color = [cm[col_map](val) for val in hist_edges]
    
    axes["A"].bar(
        hist_edges[:-1] + 0.5 * (hist_edges[1]-hist_edges[0]),
        hist,
        color=color,
        width=hist_edges[1]-hist_edges[0])
    
    if labels is False:
        # Remove the y-axis
        axes["A"].set_yticks([])
    else:
        axes["A"].set_ylabel("Density", fontsize=20)
        axes["A"].set_xlabel("Dose", fontsize=20)
        
    # Adjust margins
    fig.set_size_inches(w,h)
    
    plt.savefig(file_name)

def dr_plot(
    x: np.ndarray,
    d: np.ndarray,
    t: np.ndarray,
    gt: Callable,
    w: float,
    h: float,
    samples: int = None,
    file_name: str = "dr_plot.pdf",
):
    col_map = "viridis"
    
    # Plotting points
    plotting_samples = np.linspace(0.001, 0.999, 65)
        
    # Set number of samples
    if samples is None:
        samples = len(d)
    else:
        samples = min(samples, len(d))
        
    # Sample
    ids = np.random.choice(len(d), samples, replace=False)
    
    fig, axes = plt.subplot_mosaic("A")
    
    # Plot curves
    for id in ids:
        axes["A"].plot(
            plotting_samples,
            [gt(x[id].reshape(1,-1), help_d, t[id]).item() for help_d in plotting_samples],
            alpha=0.25,
            color=cm[col_map](d[id])
        )
    
    # Plot factuals
    for id in ids:
            axes["A"].plot(
                d[id], 
                gt(x[id].reshape(1,-1), d[id], t[id]).item(), 
                'o', 
                color='black', 
                markersize=0.25,
                alpha=0.5)
    
    axes["A"].set_ylabel("Outcome", fontsize=20)
    axes["A"].set_xlabel("Dose", fontsize=20)
    
     # Remove the y-axis
    # axes["A"].set_xticks([])
    
    # Adjust margins
    fig.set_size_inches(w,h)
    
    plt.savefig(file_name)

def dose_dr_plot(
    x: np.ndarray,
    d: np.ndarray,
    t: np.ndarray,
    gt: Callable,
    w: float,
    h: float,
    samples: int = None,
    num_bins: int = 33,
    plot_factuals: bool = True,
    file_name: str = "ddr_plot.pdf",
):
    col_map = "viridis"
    
    # Plotting points
    plotting_samples = np.linspace(0.001, 0.999, 65)
        
    # Set number of samples
    if samples is None:
        samples = len(d)
    else:
        samples = min(samples, len(d))
        
    # Sample
    ids = np.random.choice(len(d), samples, replace=False)
    
    fig, axes = plt.subplot_mosaic("AAAA;AAAA;BBBB")
    
    # DR plot
    
    axes["A"].set_ylabel("Outcome")
    
    # Plot curves
    for id in ids:
        axes["A"].plot(
            plotting_samples,
            [gt(x[id].reshape(1,-1), help_d, t[id]).item() for help_d in plotting_samples],
            alpha=0.25,
            color=cm[col_map](d[id])
        )
    
    # Plot factuals
    if plot_factuals:
        for id in ids:
                axes["A"].plot(
                    d[id], 
                    gt(x[id].reshape(1,-1), d[id], t[id]).item(), 
                    'o', 
                    color='black', 
                    markersize=1,
                    alpha=0.5)
        
    # Remove the y-axis
    axes["A"].set_xticks([])
    
    # Dose plot
    # Get histogram
    hist, hist_edges = np.histogram(d, num_bins, density=True, range=(0,1))
    color = [cm[col_map](val) for val in hist_edges]
    
    axes["B"].bar(
        hist_edges[:-1] + 0.5 * (hist_edges[1]-hist_edges[0]),
        hist,
        color=color,
        width=hist_edges[1]-hist_edges[0])
    
    axes["B"].set_ylabel("Density")
    axes["B"].set_xlabel("Dose")
    
    # Align y-axis labels
    axes["A"].get_yaxis().set_label_coords(-0.1,0.5)
    axes["B"].get_yaxis().set_label_coords(-0.1,0.5)
    
    # Adjust margins
    fig.set_size_inches(w,h)
    
    plt.savefig(file_name)