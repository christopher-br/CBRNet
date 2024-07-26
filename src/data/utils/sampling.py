# LOAD MODULES
# Standard library
from typing import Union, Tuple

# Third party
from sklearn.model_selection import train_test_split
import numpy as np

# FUNCTIONS
def train_val_test_ids(
    num_obs: int,
    train_share: float = 0.7,
    val_share: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the range of observations into training, validation, and test sets.

    Parameters:
    num_obs (int): The total number of observations.
    train_share (float, optional): The proportion of the data to include in the training set. Defaults to 0.7.
    val_share (float, optional): The proportion of the data to include in the validation set. Defaults to 0.1.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    tuple: A tuple containing three numpy arrays. The first array contains the indices for the training set, the second contains the indices for the validation set, and the third contains the indices for the test set.
    """
    # Split in train val test
    ids = np.array(range(num_obs))
    num_train = int(train_share * num_obs)
    num_val = int(val_share * num_obs)
    
    rest, ids_train = train_test_split(ids, test_size=num_train, random_state=seed)
    ids_test, ids_val = train_test_split(rest, test_size=num_val, random_state=seed)
    
    return ids_train, ids_val, ids_test

def sample_rows(
    arr: np.ndarray, 
    num_rows: Union[int, float], 
    replace: bool = False, 
    seed: int = 42
) -> np.ndarray:
    """
    Samples a specified number of rows from a numpy array.

    Parameters:
    arr (np.ndarray): The input numpy array.
    num_rows (Union[int, float]): The number of rows to sample or the percentage of rows to sample.
    replace (bool, optional): Whether to allow sampling of the same row more than once. Defaults to False.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    np.ndarray: A numpy array containing the sampled rows.
    """
    np.random.seed(seed)
    
    if isinstance(num_rows, float):
        assert 0.0 <= num_rows <= 1.0, "Percentage must be between 0.0 and 1.0"
        num_rows = int(arr.shape[0] * num_rows)
    
    row_indices = np.random.choice(arr.shape[0], size=num_rows, replace=replace)
    
    return arr[row_indices]

def sample_columns(
    arr: np.ndarray, 
    num_cols: Union[int, float], 
    replace: bool = False, 
    seed: int = 42
) -> np.ndarray:
    """
    Samples a specified number of columns from a numpy array.

    Parameters:
    arr (np.ndarray): The input numpy array.
    num_cols (Union[int, float]): The number of columns to sample or the percentage of columns to sample.
    replace (bool, optional): Whether to allow sampling of the same column more than once. Defaults to False.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    np.ndarray: A numpy array containing the sampled columns.
    """
    np.random.seed(seed)
    
    if isinstance(num_cols, float):
        assert 0.0 <= num_cols <= 1.0, "Percentage must be between 0.0 and 1.0"
        num_cols = int(arr.shape[1] * num_cols)
    
    col_indices = np.random.choice(arr.shape[1], size=num_cols, replace=replace)
    
    return arr[:,col_indices]
