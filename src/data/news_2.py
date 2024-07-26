# Source: https://github.com/lushleaf/varying-coefficient-net-with-functional-tr

# LOAD MODULES
# Standard library
from typing import Union, Optional, Dict, Any

# Proprietary
from src.data.utils import train_val_test_ids, sample_rows, normalize, ContinuousData

# Third party
from sklearn.utils import Bunch
from sklearn.neighbors import KernelDensity
import numpy as np

# CUSTOM FUNCTIONS
def sample_doses(
    matrix: np.ndarray, 
    weights: np.ndarray, 
    alpha: float = 2,
) -> np.ndarray:
    """
    Samples doses based on a given matrix, weights, and alpha parameter.

    The function calculates alpha and beta parameters for a beta distribution, and then samples doses from this distribution.

    Parameters:
    matrix (np.ndarray): The 2D numpy array used to calculate the alpha and beta parameters.
    weights (np.ndarray): The 1D numpy array used to calculate the alpha and beta parameters.
    alpha (float): The alpha parameter for the beta distribution. Defaults to 2.

    Returns:
    np.ndarray: The sampled doses, represented as a 1D numpy array.
    """
    tt = np.sum(weights[2] * matrix, axis=1) / (2. * np.sum(weights[1] * matrix, axis=1))
    beta = (alpha - 1) / tt + 2 - alpha
    beta = np.abs(beta) + 0.0001
    doses = np.random.beta(alpha, beta)
    
    return doses

def response(
    matrix: np.ndarray, 
    doses: np.ndarray, 
    weights: np.ndarray, 
    **kwargs: Dict[str, Any]
) -> np.ndarray:
    """
    Calculates the response variable for a given matrix, doses, and weights.

    The function calculates the response variable based on a complex formula involving the doses, weights, and several columns of the matrix. The formula involves trigonometric functions, exponential functions, and hyperbolic functions.

    Parameters:
    matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
    doses (np.ndarray): A 1D numpy array representing the doses for each observation.
    weights (np.ndarray): A 1D numpy array representing the weights for each observation.
    **kwargs: Dummy to improve compatibility with other functions.

    Returns:
    np.ndarray: A 1D numpy array representing the calculated response for each observation.
    """
    res1 = np.maximum(
        -2, np.minimum(
            2, np.exp(
                0.3 * (
                    (3.14159 * np.sum(weights[1] * matrix, axis=1) / np.sum(weights[2] * matrix, axis=1)) - 1
                )
            ))
    )
    res2 = 20. * (
        np.sum(weights[0] * matrix, axis=1)
    )
    y = 2 * (4 * (doses - 0.5)**2 * np.sin(0.5 * 3.14159 * doses)) * (res1 + res2)
    return y

# DATA LOADING FUNCTION
def load_data(
    data_path: str = "data/News-2.npy",
    sample_size: Optional[Union[int, float]] = None,
    bias: float = 2.,
    train_share: float = 0.7,
    val_share: float = 0.1,
    seed: int = 42,
    noise_outcome: float = 0.5,
    rm_confounding: bool = False,
    x_resampling: bool = False,
) -> Bunch:
    """
    Loads and processes data for a machine learning experiment.

    The function loads a dataset from a file, optionally samples a subset of the data, normalizes the data, generates weight vectors, doses, and outcomes, and splits the data into training, validation, and test sets.

    Parameters:
    data_path (str): The path to the .npy file containing the dataset. Defaults to "continuous/data/datasets/News-2.npy".
    bias (float): Tunable bias for sampling doses. 1 for no bias. Defaults to 2.
    sample_size (Optional[Union[int, float]]): The number or proportion of rows to sample from the dataset. If None, all rows are used. Defaults to None.
    train_share (float): The proportion of the dataset to use for the training set. Defaults to 0.7.
    val_share (float): The proportion of the dataset to use for the validation set. Defaults to 0.1.
    seed (int): The seed for the random number generator. Defaults to 42.
    noise_outcome (float): The standard deviation of the Gaussian noise added to the outcome. Defaults to 0.5.

    Returns:
    Bunch: A Bunch object containing the processed data and metadata.
    """
    # Load raw data
    matrix = np.load(data_path)
    
    # Sample rows if sample_size is specified
    if sample_size is not None:
        matrix = sample_rows(matrix, sample_size, seed=seed)
    
    # Normalize
    matrix = normalize(matrix)

    # Save info
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]
    
    # Resample x from uniform distribution per variable
    if x_resampling:
        min_vals = matrix.min(axis=0)
        max_vals = matrix.max(axis=0)
        # Continuous variables
        for i in range(num_cols):
            matrix[:,i] = np.random.uniform(min_vals[i], max_vals[i], num_rows)
    
    # Set seed
    np.random.seed(seed)
    
    # Generate weight vectors
    weights = np.random.randn(3, num_cols)
    for idx in range(3):
        sum_of_sq_row = np.sum(weights[idx,:] ** 2)
        weights[idx,:] = weights[idx,:] / np.sqrt(sum_of_sq_row)
    
    # Generate doses
    doses = sample_doses(
        matrix=matrix,
        weights=weights,
        alpha=bias,
    )

    # Remove confounding if necessary
    if rm_confounding:
        np.random.shuffle(doses)

    # Define outcome function
    def get_outcome(
        matrix: np.ndarray, 
        doses: np.ndarray, 
        treatments: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the outcome for a given matrix and doses.

        The function calls the `response` function to calculate the outcome based on the matrix and doses.

        Parameters:
        matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
        doses (np.ndarray): A 1D numpy array representing the doses for each observation.
        **kwargs: Dummy to improve compatibility with other functions.

        Returns:
        np.ndarray: A 1D numpy array representing the calculated outcome for each observation.
        """
        y = response(matrix, doses, weights)
        return y
    
    # Generate outcomes
    y = get_outcome(matrix,doses,None)
    # Assign error
    y = y + np.random.randn(num_rows) * np.sqrt(noise_outcome)
    
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