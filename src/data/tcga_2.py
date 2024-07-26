# Source: https://github.com/ioanabica/SCIGAN/blob/main/data_simulation.py

# LOAD MODULES
# Standard library
from typing import Union, Optional

# Proprietary
from src.data.utils import (
    train_val_test_ids,
    sample_rows,
    sample_columns,
    softmax,
    get_beta,
    ContinuousData,
)

# Third party
from sklearn.utils import Bunch
from sklearn.neighbors import KernelDensity
import numpy as np


# CUSTOM FUNCTIONS
def response_0(x: np.ndarray, d: np.ndarray, w_response: np.ndarray) -> np.ndarray:
    """
    Computes a response based on weights, covariates, and a dose.

    The function calculates the response based on the input features, dose, and response weights. The calculation involves a complex mathematical expression that uses various elements of the input features and the dose.

    Parameters:
    x (np.ndarray): The input features, represented as a 1D numpy array.
    d (np.ndarray): The doses.
    w_response (np.ndarray): The response weights, represented as a 2D numpy array.

    Returns:
    np.ndarray: The calculated response.
    """
    in0 = np.dot(x, w_response[0][0])
    in1 = np.dot(x, w_response[0][1])
    in2 = np.dot(x, w_response[0][2])

    y = 10.0 * (in0 + 12.0 * d * (d - 0.75 * (in1 / in2)) ** 2)

    return y


def response_1(x: np.ndarray, d: np.ndarray, w_response: np.ndarray) -> np.ndarray:
    """
    Computes a response based on weights, covariates, and a dose.

    The function calculates the response based on the input features, dose, and response weights. The calculation involves a complex mathematical expression that uses various elements of the input features and the dose.

    Parameters:
    x (np.ndarray): The input features, represented as a 1D numpy array.
    d (np.ndarray): The doses.
    w_response (np.ndarray): The response weights, represented as a 2D numpy array.

    Returns:
    np.ndarray: The calculated response.
    """
    in0 = np.dot(x, w_response[1][0])
    in1 = np.dot(x, w_response[1][1])
    in2 = np.dot(x, w_response[1][2])

    y = 10.0 * (in0 + np.sin(np.pi * (in1 / in2) * d))

    return y


def response_2(x: np.ndarray, d: np.ndarray, w_response: np.ndarray) -> np.ndarray:
    """
    Computes a response based on weights, covariates, and a dose.

    The function calculates the response based on the input features, dose, and response weights. The calculation involves a complex mathematical expression that uses various elements of the input features and the dose.

    Parameters:
    x (np.ndarray): The input features, represented as a 1D numpy array.
    d (np.ndarray): The doses.
    w_response (np.ndarray): The response weights, represented as a 2D numpy array.

    Returns:
    np.ndarray: The calculated response.
    """
    in0 = np.dot(x, w_response[2][0])
    in1 = np.dot(x, w_response[2][1])
    in2 = np.dot(x, w_response[2][2])

    y = 10.0 * (in0 + 12.0 * (in1 * d - in2 * d**2))

    return y


def load_data(
    data_path: str = "data/TCGA-1.csv",
    sample_size: Optional[Union[int, float]] = None,
    num_covariates: Optional[Union[int, float]] = None,
    num_treatments: int = 1,
    treatment_bias: float = 2.0,
    dose_bias: float = 2.0,
    train_share: float = 0.7,
    val_share: float = 0.1,
    seed: int = 3,
    noise_outcome: float = 0.2,
    x_resampling: bool = False,
    rm_confounding_t: bool = False,
    rm_confounding_d: bool = False,
) -> Bunch:
    """
    Loads and processes the TCGA-S-2 dataset for a machine learning experiment.

    The function loads the dataset from the specified path, applies a bias to the treatments and doses, splits the data into training, validation, and test sets, and adds Gaussian noise to the outcomes.

    Parameters:
    data_path (str): The path to the dataset. Defaults to "continuous/data/datasets/TCGA-1.csv".
    sample_size (Optional[Union[int, float]]): The sample size to use for the dataset. If None, the full dataset is used. Defaults to None.
    num_covariates (Optional[Union[int, float]]): The number of covariates in the dataset. If None, all covariates are used. Defaults to None.
    num_treatments (int): The number of treatments in the dataset. Defaults to 3.
    treatment_bias (float): The bias to apply to the treatments. Defaults to 3.0.
    dose_bias (float): The bias to apply to the doses. Defaults to 3.0.
    train_share (float): The proportion of the dataset to use for the training set. Defaults to 0.7.
    val_share (float): The proportion of the dataset to use for the validation set. Defaults to 0.1.
    seed (int): The seed for the random number generator. Defaults to 3.
    noise_outcome (float): The standard deviation of the Gaussian noise added to the outcomes. Defaults to 0.2.

    Returns:
    Bunch: A Bunch object containing the processed data and metadata.
    """
    # Define eps
    eps = np.finfo(float).eps

    # Set seed
    np.random.seed(seed)

    # Num treatments
    num_weights = 3

    # Load raw data
    matrix = np.loadtxt(data_path, delimiter=",", skiprows=1)

    # Sample rows and columns
    if sample_size is not None:
        matrix = sample_rows(matrix, sample_size)
    if num_covariates is not None:
        matrix = sample_columns(matrix, num_covariates)

    # Save info
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]

    # Standardize
    matrix = (matrix - np.min(matrix, axis=0)) / (
        eps + (np.max(matrix, axis=0) - np.min(matrix, axis=0))
    )

    # Normalize every observation
    for i in range(num_rows):
        matrix[i] = matrix[i] / np.linalg.norm(matrix[i])

    # Resample x from uniform distribution per variable
    if x_resampling:
        min_vals = matrix.min(axis=0)
        max_vals = matrix.max(axis=0)
        # Continuous variables
        for i in range(num_cols):
            matrix[:,i] = np.random.uniform(min_vals[i], max_vals[i], num_rows)

    # Get weights
    w_response = np.zeros((3, num_weights, num_cols))
    for i in range(num_treatments):
        for j in range(num_weights):
            w_response[i][j] = np.random.uniform(0, 10, size=(num_cols))
            w_response[i][j] = w_response[i][j] / np.linalg.norm(w_response[i][j])

    # Sample doses and outcomes
    doses = np.zeros(num_rows)
    outcomes = np.zeros(num_rows)
    treatments = np.zeros(num_rows).astype(int)

    # Iterate per row
    for idx, x in enumerate(matrix):
        potential_doses = np.zeros(num_treatments)
        potential_outcomes = np.zeros(num_treatments)
        for treatment in range(num_treatments):
            if treatment == 0:
                # Get optimal dose
                b = (
                    0.75
                    * np.dot(x, w_response[treatment][1])
                    / np.dot(x, w_response[treatment][2])
                )

                if b >= 0.75:
                    optimal_dose = b / 3.0
                else:
                    optimal_dose = 1.0

                alpha = dose_bias
                dose_sample = np.random.beta(alpha, get_beta(alpha, optimal_dose))

                # Get outcome
                outcome_sample = response_0(x, dose_sample, w_response)

                # Append
                potential_doses[treatment] = dose_sample
                potential_outcomes[treatment] = outcome_sample

            elif treatment == 1:
                # Get optimal dose
                optimal_dose = np.dot(x, w_response[treatment][2]) / (
                    2.0 * np.dot(x, w_response[treatment][1])
                )

                alpha = dose_bias
                dose_sample = np.random.beta(alpha, get_beta(alpha, optimal_dose))

                if optimal_dose <= 0.001:
                    dose_sample = 1 - dose_sample

                # Get outcome
                outcome_sample = response_1(x, dose_sample, w_response)

                # Append
                potential_doses[treatment] = dose_sample
                potential_outcomes[treatment] = outcome_sample

            else:
                # Get optimal dose
                optimal_dose = np.dot(x, w_response[treatment][1]) / (
                    2.0 * np.dot(x, w_response[treatment][2])
                )

                alpha = dose_bias
                dose_sample = np.random.beta(alpha, get_beta(alpha, optimal_dose))

                if optimal_dose <= 0.001:
                    dose_sample = 1 - dose_sample

                # Get outcome
                outcome_sample = response_2(x, dose_sample, w_response)

                # Append
                potential_doses[treatment] = dose_sample
                potential_outcomes[treatment] = outcome_sample

        treatment_coeffs = [
            treatment_bias * (potential_outcomes[i] / np.max(potential_outcomes))
            for i in range(num_treatments)
        ]
        treatment = np.random.choice(num_treatments, p=softmax(treatment_coeffs))

        doses[idx] = potential_doses[treatment]
        outcomes[idx] = potential_outcomes[treatment] + np.random.normal(
            0, noise_outcome
        )
        treatments[idx] = int(treatment)

    def get_outcome(
        matrix: np.ndarray,
        doses: np.ndarray,
        treatments: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the outcome for a given matrix, doses, and treatments.

        The function calculates the outcome based on the matrix, doses, and treatments. The calculation involves calling response functions for each treatment and selecting the appropriate outcome based on the treatment.

        Parameters:
        matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
        doses (np.ndarray): A 1D numpy array representing the doses for each observation.
        treatments (optional, np.ndarray): A 1D numpy array representing the treatment for each observation. If not passed, assumes treatment to be 0 for all observations. Defaults to None.
        **kwargs: Dummy to improve compatibility with other functions.

        Returns:
        np.ndarray: A 1D numpy array representing the calculated outcome for each observation.
        """
        num_rows = matrix.shape[0]

        potential_outcomes = np.zeros((num_rows, 3))

        potential_outcomes[:, 0] = response_0(matrix, doses, w_response)
        potential_outcomes[:, 1] = response_1(matrix, doses, w_response)
        potential_outcomes[:, 2] = response_2(matrix, doses, w_response)

        y = potential_outcomes[np.arange(matrix.shape[0]), treatments.astype(int)]
        return y
    
    # Remove treatment confounding if necessary
    if rm_confounding_t:
        np.random.shuffle(treatments)

        # Get outcomes
        outcomes = get_outcome(matrix, doses, treatments)+ np.random.normal(
            0, 
            noise_outcome, 
            num_rows
        )

    # Remove dose confounding if necessary
    if rm_confounding_d:
        # Shuffel d per treatment
        for i in range(num_treatments):
            np.random.shuffle(doses[treatments == i])

        # Get outcomes
        outcomes = get_outcome(matrix, doses, treatments)+ np.random.normal(
            0, 
            noise_outcome, 
            num_rows
        )

    # Get train/val/test ids
    train_ids, val_ids, test_ids = train_val_test_ids(
        num_rows,
        train_share=train_share,
        val_share=val_share,
    )

    # Generate bunch
    data = ContinuousData(
        x=matrix,
        t=treatments,
        d=doses,
        y=outcomes,
        ground_truth=get_outcome,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    return data
