# Source: https://github.com/lushleaf/varying-coefficient-net-with-functional-tr

# LOAD MODULES
# Standard library
...

# Proprietary
from src.data.utils import train_val_test_ids, ContinuousData

# Third party
from sklearn.utils import Bunch
from sklearn.neighbors import KernelDensity
import numpy as np

# DATA LOADING FUNCTION
def load_data(
    num_rows: int = 700,
    num_var: int = 6,
    bias = 1.,
    train_share: float = 0.7,
    val_share: float = 0.1,
    seed: int = 5,
    noise_dose: float = 0.5,
    noise_outcome: float = 0.5,
    rm_confounding: bool = False,
    x_resampling: bool = False,
) -> Bunch:
    """
    Generates and processes synthetic data for a machine learning experiment.

    The function generates a covariate matrix, samples doses, calculates outcomes, and splits the data into training, validation, and test sets.

    Parameters:
    num_rows (int): The number of observations to generate. Defaults to 700.
    num_var (int): The number of variables to generate. Defaults to 6.
    bias (float): The proportion of the doses that are generated from confounded formula. Rest is randomly sampled from [0,1]. Defaults to 1.
    train_share (float): The proportion of the dataset to use for the training set. Defaults to 0.7.
    val_share (float): The proportion of the dataset to use for the validation set. Defaults to 0.1.
    seed (int): The seed for the random number generator. Defaults to 5.
    noise_dose (float): The standard deviation of the Gaussian noise added to the doses. Defaults to 0.5.
    noise_outcome (float): The standard deviation of the Gaussian noise added to the outcomes. Defaults to 0.5.

    Returns:
    Bunch: A Bunch object containing the processed data and metadata.
    """
    # Set seed
    np.random.seed(seed)
    
    # Generate covariate matrix
    matrix = np.random.rand(num_rows,num_var)
    
    # Resample x from uniform distribution per variable
    if x_resampling:
        min_vals = matrix.min(axis=0)
        max_vals = matrix.max(axis=0)
        # Continuous variables
        for i in range(num_var):
            matrix[:,i] = np.random.uniform(min_vals[i], max_vals[i], num_rows)
    
    # Sample doses
    x1 = matrix[:,0]
    x2 = matrix[:,1]
    x3 = matrix[:,2]
    x4 = matrix[:,3]
    x5 = matrix[:,4]
    
    # Sample doses
    doses = (10. * np.sin(np.maximum(np.maximum(x1, x2), x3)) + np.maximum(np.maximum(x3, x4), x5)**3)/ \
        (1. + (x1 + x5)**2) + \
            np.sin(0.5 * x3) * (1. + np.exp(x4 - 0.5 * x3)) + \
                x3**2 + 2. * np.sin(x4) + 2.*x5 - 6.5
    
    # Add noise
    doses = doses + np.random.randn(num_rows) * noise_dose
    
    # Add tunable bias
    # Proprietart addition to code
    doses = bias * doses + (1. - bias) * np.random.logistic(scale=1.,size=num_rows)
    
    # Link
    doses = (1. / (1. + np.exp(-1. * doses)))
    
    # Remove confounding if necessary
    if rm_confounding:
        np.random.shuffle(doses)
    
    # Calculate outcome
    def get_outcome(
        matrix: np.ndarray, 
        doses: np.ndarray, 
        treatments: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the outcome for a given matrix and doses.

        The function calculates the outcome based on the matrix and doses. The calculation involves a complex mathematical expression that uses various elements of the matrix and the doses.

        Parameters:
        matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
        doses (np.ndarray): A 1D numpy array representing the doses for each observation.
        **kwargs: Dummy to improve compatibility with other functions.

        Returns:
        np.ndarray: A 1D numpy array representing the calculated outcome for each observation.
        """
        x1 = matrix[:,0]
        x3 = matrix[:,2]
        x4 = matrix[:,3]
        x6 = matrix[:,5]
        
        y = np.cos((doses - 0.5) * 3.14159 * 2.) * (doses**2 + (4.*np.maximum(x1, x6)**3)/(1. + 2.*x3**2)*np.sin(x4))
        
        return y
    
    y = get_outcome(matrix, doses, None)
    
    # Add noise
    y = y + np.random.randn(num_rows) * noise_outcome
    
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