# Source: https://github.com/lushleaf/varying-coefficient-net-with-functional-tr

# LOAD MODULES
# Standard library
from typing import Union, Optional

# Proprietary
from src.data.utils import train_val_test_ids, sample_rows, ContinuousData

# Third party
from sklearn.utils import Bunch
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd

# DATA LOADING FUNCTION
def load_data(
    data_path: str = "data/IHDP-1.csv",
    bias: float = 1.,
    sample_size: Optional[Union[int, float]] = None,
    train_share: float = 0.7,
    val_share: float = 0.1,
    seed: int = 5,
    alpha: float = 5.0,
    factor1: float = 1.0,
    factor2: float = 1.0,
    noise_dose: float = 0.5,
    noise_outcome: float = 0.5,
    rm_confounding: bool = False,
    x_resampling: bool = False,
) -> Bunch:
    """
    Loads and preprocesses data from a CSV file.

    The function loads the data, optionally samples a subset of rows, normalizes the data, calculates an outcome variable, adds noise to the outcome, and splits the data into training, validation, and test sets.

    Parameters:
    data_path (str): Path to the CSV file to load. Defaults to "data/datasets/IHDP-1.csv".
    bias (float): The proportion of the doses that are generated from confounded formula. Rest is randomly sampled from [0,1]. Defaults to 1.
    sample_size (int, float): Number or proportion of rows to sample from the data. If None, all rows are used. Defaults to None.
    train_share (float): Proportion of the data to include in the training set. Defaults to 0.7.
    val_share (float): Proportion of the data to include in the validation set. Defaults to 0.1.
    seed (int): Seed for the random number generator. Defaults to 42.
    alpha (float): Parameter for the outcome calculation. Defaults to 5.0.
    factor1 (float): First factor for the outcome calculation. Defaults to 1.0. Alternatively v1: 0.5 or v2: 1.5
    factor2 (float): Second factor for the outcome calculation. Defaults to 1.0. Alternatively v1: 1.5 or v2: 0.5
    noise_dose (float): Standard deviation of the noise to add to the doses. Defaults to 0.5.
    noise_outcome (float): Standard deviation of the noise to add to the outcome. Defaults to 0.5.
    rm_confounding (bool): Whether to remove confounding from the doses. Defaults to False.
    x_resampling (bool): Whether to resample x from uniform distribution per variable. Defaults to False.

    Returns:
    Bunch: A Bunch object containing the preprocessed data, the outcome variable, the ground truth function, and the indices for the training, validation, and test sets.
    """
    # Set seed
    np.random.seed(seed)
    
    # Load raw data
    matrix = pd.read_csv(data_path, sep=",", header=0)
    
    # To numpy
    matrix = matrix.to_numpy()
    
    # Sample rows if sample_size is specified
    if sample_size is not None:
        matrix = sample_rows(matrix, sample_size, seed=seed)
    
    # Drop columns
    # NOTE: Only reading 25 columns of dataframe as in original script. 3 more available
    matrix = matrix[:,2:27]
    
    # Save info
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]
    
    # Normalize
    for col in range(num_cols):
        minval = min(matrix[:, col]) * 1.
        maxval = max(matrix[:, col]) * 1.
        matrix[:, col] = (1. * (matrix[:, col] - minval))/maxval
    
    # Resample x from uniform distribution per variable
    if x_resampling:
        min_vals = matrix.min(axis=0)
        max_vals = matrix.max(axis=0)
        # Continuous variables
        for i in range(0,6):
            matrix[:,i] = np.random.uniform(min_vals[i], max_vals[i], num_rows)
        # Binary variables
        for i in range(6,25):
            matrix[:,i] = np.random.binomial(1, 0.5, num_rows)
    
    # Save continuous variables
    x1 = matrix[:,0]
    x2 = matrix[:,1]
    x3 = matrix[:,2]
    x4 = matrix[:,4]
    x5 = matrix[:,5]
    
    # Calc cate_mean
    cate_idx1 = np.array([3,6,7,8,9,10,11,12,13,14])
    cate_idx2 = np.array([15,16,17,18,19,20,21,22,23,24])
    
    cate_mean1 = np.mean(np.mean(matrix[:,cate_idx1], axis=1))
    cate_mean2 = np.mean(np.mean(matrix[:,cate_idx1], axis=1)) # Taken from og github. Hypo: Should be cate_idx2
    
    doses = x1/(1. + x2) + \
        np.maximum(np.maximum(x3, x4), x5)/(0.2 + np.minimum(np.minimum(x3, x4), x5)) + \
            np.tanh((np.sum(matrix[:,cate_idx2], axis=1)/10. - cate_mean2) * alpha) \
                - 2.
    
    # Add error
    doses = doses + np.random.randn(matrix.shape[0]) * noise_dose
    
    # Add tunable bias
    # Proprietart addition to code
    doses = bias * doses + (1. - bias) * np.random.logistic(scale=1.,size=num_rows) / 2.
    
    # Transform according to link function
    doses = 1. / (1. + np.exp(-2. * doses))
    
    # Remove confounding if necessary
    if rm_confounding:
        np.random.shuffle(doses)

    def get_outcome(
        matrix: np.ndarray, 
        doses: np.ndarray, 
        treatments: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the outcome variable for a given dataset and doses.

        The function calculates the outcome variable based on a complex formula involving the doses and several columns of the dataset. The formula involves trigonometric functions, exponential functions, and hyperbolic functions.

        Parameters:
        matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
        doses (np.ndarray): A 1D numpy array representing the doses for each observation.
        **kwargs: Dummy to improve compatibility with other functions.

        Returns:
        np.ndarray: A 1D numpy array representing the calculated outcome for each observation.
        """
        # Only take continuous variables
        x1 = matrix[:,0]
        x2 = matrix[:,1]
        x3 = matrix[:,2]
        x4 = matrix[:,4]
        x5 = matrix[:,5]
        
        # Calc outcome
        y = 1. / (1.2 - doses) * np.sin(doses * 3. * 3.14159) * (
            factor1 * np.tanh((np.sum(matrix[:, cate_idx1], axis=1) / 10. - cate_mean1) * alpha) + \
            factor2 * np.exp(0.2 * (x1 - x5)) / (0.1 + np.minimum(x2, x3, x4))
        )
        
        return y
    
    # Sample outcomes
    y = get_outcome(matrix, doses, None)
    
    # Add noise
    y = y + np.random.randn(matrix.shape[0]) * noise_outcome
    
    # Generate a dummy array for treatment
    t = np.zeros(num_rows).astype(int)
    
    # Get train/val/test ids
    train_ids, val_ids, test_ids = train_val_test_ids(num_rows,
                                                      train_share=train_share,
                                                      val_share=val_share,)
    
    # Generate bunch
    data = ContinuousData(
        x = matrix,
        t = t,
        d = doses,
        y = y,
        ground_truth = get_outcome,
        train_ids = train_ids,
        val_ids = val_ids,
        test_ids = test_ids,
    )
    
    return data