# LOAD MODULES
# Standard library
...

# Third party
import numpy as np

# FUNCTIONS
def normalize(
    matrix: np.ndarray
) -> np.ndarray:
    """
    Normalizes a 2D numpy array column-wise.

    The function divides each column by its maximum value, resulting in all values in the column being in the range [0, 1].

    Parameters:
    matrix (np.ndarray): The 2D numpy array to normalize.

    Returns:
    np.ndarray: The normalized 2D numpy array.
    """
    num_cols = matrix.shape[1]
    
    for idx in range(num_cols):
        max_value = max(matrix[:, idx])
        matrix[:, idx] = matrix[:, idx] / max_value
    
    return matrix

def softmax(
    x: np.ndarray,
) -> np.ndarray:
    """
    Computes the softmax of a numpy array.

    The softmax function is a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers.

    Parameters:
    x (np.ndarray): A numpy array for which to compute the softmax.

    Returns:
    np.ndarray: A numpy array representing the softmax of the input.
    """
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def get_beta(
    alpha: float, mode: float
) -> float:
    """
    Computes the beta value based on the provided alpha and mode.

    The function calculates the beta value using a specific formula. If the mode is less than or equal to 0.001 or greater than or equal to 1.0, beta is set to 1.0. Otherwise, beta is calculated using the formula.

    Parameters:
    alpha (float): The alpha value used in the calculation.
    mode (float): The mode value used in the calculation.

    Returns:
    float: The calculated beta value.
    """
    if (mode <= 0.001 or mode >= 1.0):
        beta = 1.0
    else:
        beta = (alpha - 1.0) / float(mode) + (2.0 - alpha)

    return beta
