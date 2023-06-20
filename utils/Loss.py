# Code inspired by https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py

import numpy as np
import torch

def mmd_lin(hidden, x, d, binner, n_cluster, **kwargs):
    """
    Calculates the linear MMD divergence of observations from one head to all other observations.
    
    Parameters:
        hidden (torch tensor): All hidden representations of observations
        x (torch tensor): All untransformed observations
        d (torch tensor): Dosages assigned to observations
        n_cluster (int): Number of clusters found in the data
        weight_v (torch tensor): Weight vector over all heads, i.e., relative frequency
    """
    # Get the binned observations
    obs_binned = binner(d=d, x=x)
    
    # Ini mmd
    mmd = 0
    
    # Calculate means and divergence
    for i in range(n_cluster):
        # Catch error if no observation for head i, and assign no loss in that case
        if hidden[obs_binned == i].shape[0] <= 1 or hidden[obs_binned != i].shape[0] <= 1:
            mmd = mmd
        else:
            # Means of obs with head i
            means = torch.mean(hidden[obs_binned == i], 0)
            # Get means of complements
            means_comp = torch.mean(hidden, 0)
            # Increment mdd by sum of sqaured divergences multiplied by weight of cluster
            mmd += torch.sqrt(torch.sum(torch.square(means - means_comp))) # TODO: Reweight here to equalize
        
    # Calculate average
    mmd = mmd / n_cluster
        
    return mmd

def mmd_lin_w(hidden, x, d, binner, n_cluster, weight_v, **kwargs):
    """
    Calculates the weighted linear MMD divergence of observations from one head to all other observations.
    Weighting of MMD divergence is to overcome dosage selection bias.
    
    Parameters:
        hidden (torch tensor): All hidden representations of observations
        x (torch tensor): All untransformed observations
        d (torch tensor): Dosages assigned to observations
        n_cluster (int): Number of clusters found in the data
        weight_v (torch tensor): Weight vector over all heads, i.e., relative frequency
    """
    obs_binned = binner(d=d, x=x)
    
    # Get observation weights
    obs_weight = torch.tensor([weight_v[i] for i in obs_binned]).reshape(-1,1)
    
    # Ini mmd
    mmd = 0
    
    # Calculate means and divergence
    for i in range(n_cluster):
        # Catch error if no observation for head i, and assign no loss in that case
        if hidden[obs_binned == i].shape[0] <= 1 or hidden[obs_binned != i].shape[0] <= 1:
            mmd = mmd
        else:
            # Means of obs with head i
            means = torch.mean(hidden[obs_binned == i], 0)
            # Weighted means of complements
            # Get weighted obs
            hidden_w = hidden * obs_weight # or to only use observations with bin != i: x_w = x[obs_binned != i] * obs_weight[obs_binned != i]
            # Get means
            means_comp = torch.sum(hidden_w, 0)
            means_comp = means_comp / torch.sum(obs_weight) # or to only use observations with bin != i: means_comp = means_comp / torch.sum(obs_weight[obs_binned != i])
            # Increment mdd by sum of sqaured divergences multiplied by weight of cluster
            mmd += torch.sqrt(torch.sum(torch.square(means - means_comp))) * weight_v[i]
        
    # Calculate average
    mmd = mmd / n_cluster
        
    return mmd
    
def mmd_rbf(hidden, x, d, binner, n_cluster, weight_v, sig=0.1, **kwargs):
    """
    Calculates the  RBF MMD divergence of observations from one head to all other observations.
    
    Parameters:
        hidden (torch tensor): All hidden representations of observations
        x (torch tensor): All untransformed observations
        d (torch tensor): Dosages assigned to observations
        n_cluster (int): Number of clusters found in the data
        weight_v (torch tensor): Weight vector over all heads, i.e., relative frequency
        sig (float): Sigma. Free parameter in calculating the kernel distance
        
    Inspiration taken from https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    """
    obs_binned = binner(d=d, x=x)
    
    # Ini mmd
    mmd = 0
    
    # Calculate means and divergence
    for i in range(n_cluster):
        # Catch error if no (or only one) observation for head i, and assign no loss in that case
        if hidden[obs_binned == i].shape[0] <= 1 or hidden[obs_binned != i].shape[0] <= 1:
            mmd = mmd
        else:
            # Get obs in head i
            hidden_i = hidden[obs_binned == i] 
            n_hidden_i = hidden_i.shape[0]
            # Get obs not in head i
            hidden_c = hidden # or to only use observations with bin != i: x_c = x[obs_binned != i]
            n_hidden_c = hidden_c.shape[0]
            
            # Kernel representations
            K_ii = torch.exp(-torch.cdist(hidden_i, hidden_i)/(2*(sig**2)))
            K_ic = torch.exp(-torch.cdist(hidden_i, hidden_c)/(2*(sig**2)))
            K_cc = torch.exp(-torch.cdist(hidden_c, hidden_c)/(2*(sig**2)))
            
            # Take average of kernels - Do not add diag. for K_ii and K_cc
            avg_K_ii = (torch.sum(K_ii) - n_hidden_i) / (n_hidden_i * (n_hidden_i - 1)) # subtract diagonal
            
            avg_K_ic = torch.sum(K_ic) / (n_hidden_c * n_hidden_i)
            
            avg_K_cc = (torch.sum(K_cc) - n_hidden_c) / (n_hidden_c * (n_hidden_c - 1)) # subtract diagonal
            
            # Add to joint mmd
            mmd += avg_K_ii - 2 * avg_K_ic + avg_K_cc
        
    # Calculate average
    mmd = mmd / n_cluster
        
    return mmd

def mmd_rbf_w(hidden, x, d, binner, n_cluster, weight_v, sig=0.1, **kwargs):
    """
    Calculates the weighted RBF MMD divergence of observations from one head to all other observations.
    Weighting of MMD divergence is to overcome dosage selection bias.
    
    Parameters:
        hidden (torch tensor): All hidden representations of observations
        x (torch tensor): All untransformed observations
        d (torch tensor): Dosages assigned to observations
        n_cluster (int): Number of clusters found in the data
        weight_v (torch tensor): Weight vector over all heads, i.e., relative frequency
        sig (float): Sigma. Free parameter in calculating the kernel distance
        
    Inspiration taken from https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    """
    obs_binned = binner(d=d, x=x)
    
    # Get observation weights
    obs_weight = torch.tensor([weight_v[i] for i in obs_binned]).reshape(-1,1)
    
    # Ini mmd
    mmd = 0
    
    # Calculate means and divergence
    for i in range(n_cluster):
        # Catch error if no (or only one) observation for head i, and assign no loss in that case
        if hidden[obs_binned == i].shape[0] <= 1 or hidden[obs_binned != i].shape[0] <= 1:
            mmd = mmd
        else:
            # Get obs in head i
            hidden_i = hidden[obs_binned == i] 
            n_hidden_i = hidden_i.shape[0]
            # Get obs not in head i
            hidden_c = hidden # or to only use observations with bin != i: x_c = x[obs_binned != i]
            n_hidden_c = hidden_c.shape[0]
            
            # Weight vectors
            w_v = obs_weight # or to only use observations with bin != i: w_v = obs_weight[obs_binned != i]
            
            # Kernel representations
            K_ii = torch.exp(-torch.cdist(hidden_i, hidden_i)/(2*(sig**2)))
            K_ic = torch.exp(-torch.cdist(hidden_i, hidden_c)/(2*(sig**2)))
            K_cc = torch.exp(-torch.cdist(hidden_c, hidden_c)/(2*(sig**2)))
            
            # Take weighted average of kernels - Do not add diag. for K_ii and K_cc
            avg_K_ii = (torch.sum(K_ii) - n_hidden_i) / (n_hidden_i * (n_hidden_i - 1)) # subtract diagonal
            
            w_matrix_ic = torch.ones(n_hidden_i, n_hidden_c) * w_v.t()
            avg_K_ic = torch.sum(w_matrix_ic * K_ic) / torch.sum(w_matrix_ic)
            
            w_matrix_cc = (torch.ones(n_hidden_c, n_hidden_c) * w_v * w_v.t())
            avg_K_cc = (torch.sum(w_matrix_cc * K_cc) - torch.sum((w_v)**2)) / (torch.sum(w_matrix_cc) - torch.sum((w_v)**2)) # subtract diagonal
            
            # Add to joint mmd
            mmd += avg_K_ii - 2 * avg_K_ic + avg_K_cc
        
    # Calculate average
    mmd = mmd / n_cluster
        
    return mmd

def mmd_none(hidden, x, d, binner, n_cluster, weight_v, **kwargs):
    """
    Placeholder for case when no MMD loss is calculated. Will always output 0.
    
    Parameters:
        hidden (torch tensor): All hidden representations of observations
        x (torch tensor): All untransformed observations
        d (torch tensor): Dosages assigned to observations
        n_cluster (int): Number of clusters found in the data
        weight_v (torch tensor): Weight vector over all heads, i.e., relative frequency
        sig (float): Sigma. Free parameter in calculating the kernel distance
    """
    return torch.tensor(0)

    