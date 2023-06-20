import torch
import jenkspy

from sklearn.neighbors._kde import KernelDensity
from sklearn.cluster import KMeans
from scipy.signal import argrelmin

def lin_bins(**kwargs):
    """
    Function to bin observations into equally spaced bins.
    
    Parameters:
        d (torch tensor): Dosages per observations
        n_bins (int): Number of bins to be calculated
    """
    d = kwargs['d']
    n_bins = kwargs['n_bins']
    # Get min and max dosage
    d_min = torch.min(d) - torch.finfo(torch.float16).eps
    d_max = torch.max(d) + torch.finfo(torch.float16).eps
    
    # Get bounds
    bins = torch.linspace(d_min, d_max, (n_bins + 1))
    # Change bounds
    bins[0] = -torch.inf
    bins[-1] = torch.inf
    
    # Bucketize
    obs_binned = torch.bucketize(d, bins) - 1
    
    # Define binning fct
    def binner(**kwargs):
        # Save variables
        d = kwargs['d']
        
        # Define binning logic
        binned = torch.bucketize(d, bins) - 1
        
        return binned
    
    return obs_binned, binner, n_bins
    
def quantile_bins(**kwargs):
    """
    Function to bin observations into equally sized bins (by number of contained observations).
    
    Parameters:
        d (torch tensor): Dosages per observations
        n_bins (int): Number of bins to be calculated
    """
    d = kwargs['d']
    n_bins = kwargs['n_bins']
    bins = torch.Tensor([torch.quantile(d, q) for q in torch.linspace(0., 1., (n_bins + 1))])
    # Change bounds
    bins[0] = -torch.inf
    bins[-1] = torch.inf
    
    # Bucketize
    obs_binned = torch.bucketize(d, bins) - 1
    
    # Define binning fct
    def binner(**kwargs):
        # Save variables
        d = kwargs['d']
        
        # Define binning logic
        binned = torch.bucketize(d, bins) - 1
        
        return binned
    
    return obs_binned, binner, n_bins
    
def jenks_bins(**kwargs):
    """
    Function to bin observations into bins via jenks natural break clustering.
    
    Parameters:
        d (torch tensor): Dosages per observations
        n_bins (int): Number of bins to be calculated
    """
    d = kwargs['d']
    n_bins = kwargs['n_bins']
    # Ini jenks breaks
    jnb = jenkspy.JenksNaturalBreaks(n_bins)
    # Fit
    jnb.fit(d)
    
    # Get bins
    bins = torch.Tensor(jnb.breaks_)
    
    # Change bounds
    bins[0] = -torch.inf
    bins[-1] = torch.inf
    
    # Bucketize
    obs_binned = torch.bucketize(d, bins) - 1
    
    # Define binning fct
    def binner(**kwargs):
        # Save variables
        d = kwargs['d']
        
        # Define binning logic
        binned = torch.bucketize(d, bins) - 1
        
        return binned
    
    return obs_binned, binner, n_bins

def kde_bins(**kwargs):
    """
    Function to bin observations into bins via jenks natural break clustering.
    
    Parameters:
        d (torch tensor): Dosages per observations
    """
    d = kwargs['d']
    # Get min and max dosage
    d_min = torch.min(d) - torch.finfo(torch.float16).eps
    d_max = torch.max(d) + torch.finfo(torch.float16).eps
    
    # Fit KDE
    kde = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(d.reshape(-1,1)) # for bandwidth use 'scott' or 'silverman', for kernel use 'epanechnikov' or 'gaussian'
     
    # Set evaluation values
    eval_values = torch.linspace(d_min, d_max, 100).reshape(-1,1)
    
    # Get densities of KDE over eval values
    densities = kde.score_samples(eval_values)
    
    # Get local minima
    mins = torch.Tensor(argrelmin(densities,)[0]).int().detach()
    if mins.nelement() > 0:
        bounds = torch.index_select(eval_values, 0, mins).squeeze().reshape(-1)
    else:
        bounds = mins
    
    # Get left and right bounds
    left = torch.Tensor([-torch.inf])
    right = torch.Tensor([torch.inf])
    
    # Cat
    bins = torch.cat((left, bounds, right))
    
    # Bucketize
    obs_binned = torch.bucketize(d, bins) - 1
    
    # Calc num bins
    n_bins = bins.shape[0] - 1
    
    # Define binning fct
    def binner(**kwargs):
        # Save variables
        d = kwargs['d']
        
        # Define binning logic
        binned = torch.bucketize(d, bins) - 1
        
        return binned
    
    return obs_binned, binner, n_bins

def kmeans_bins(**kwargs):
    """
    Function to bin observations into bins via kmeans clustering on x and d.
    
    Parameters:
        x (torch tensor): All observations
        d (torch tensor): Dosages per observations
        n_bins (int): Number of clusters for kmeans
    """
    x = kwargs['x']
    d = kwargs['d'].reshape(-1,1)
    n_bins = kwargs['n_bins']
    # Concat x and d
    xd = torch.cat((x, d), dim=1)
    # Init KMeans
    kmeans = KMeans(n_clusters=n_bins, init='k-means++', random_state=42, n_init='auto').fit(xd)
    
    # Get bins for training obs
    obs_binned = kmeans.labels_
    
    # Define binning fct
    def binner(**kwargs):
        # Save variables
        x = kwargs['x']
        d = kwargs['d'].reshape(-1,1)
        
        # Concat x and d
        xd = torch.cat((x, d), dim=1)
    
        # Define binning logic
        binned = kmeans.predict(xd)
        
        return binned
    
    return obs_binned, binner, n_bins