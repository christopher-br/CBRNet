def beta_mode(alpha, mode, **kwargs):
    """
    Function to calculate beta to introduce selection bias.
    Form of selection bias: All clusters have a different mode.
    
    Parameters:
        alpha (float): Strength of selection bias
        cluster_mode (float): The modal dosage
    """
    # Calculate beta
    beta = (alpha - 1.0) / mode + (2.0 - alpha)
    
    return beta