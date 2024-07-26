# LOAD MODULES
# Standard library
from typing import Callable

# Third party
import numpy as np
from scipy.integrate import romb
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

def mise_metric(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_integration_samples: int = 65,
) -> float:
    """
    Calculates the Mean Integrated Prediction Error (MIPE) for a given model and the factual treatments.

    The MIPE integrates the squared difference between the true response and the model's predicted response over all possible doses
    (1/n) * \sum{\ingral{0}{1}{(y_i(d) - y-hat_i(d))^2}dd}

    Parameters:
        x (np.ndarray): The covariates.
        response (Callable): A function representing the true response.
        model (Callable): The model to be evaluated.
        num_integration_samples (int, optional): The number of samples to be used for the integration. Defaults to 65.

    Returns:
        float: The Mean Integrated Prediction Error of the model.
    """
    # Get step size
    step_size = 1 / num_integration_samples
    num_obs = x.shape[0]

    # Generate data
    x = np.repeat(x, repeats=num_integration_samples, axis=0)
    d = np.linspace(0, 1, num_integration_samples)
    d = np.tile(d, num_obs)
    t = np.repeat(t, repeats=num_integration_samples)

    # Get true outcomes
    y = response(x, d, t)
    # Get predictions
    y_hat = model.predict(x, d, t)

    # Get mise
    mises = []
    y_chunks = y.reshape(-1, num_integration_samples)
    y_hat_chunks = y_hat.reshape(-1, num_integration_samples)

    for y_chunk, y_hat_chunk in zip(y_chunks, y_hat_chunks):
        mise = romb((y_chunk - y_hat_chunk) ** 2, dx=step_size)
        mises.append(mise)

    return np.sqrt(np.mean(mises))

def mean_dose_error(
    x: np.ndarray,
    t: np.ndarray,
    response: Callable,
    model: Callable,
    num_integration_samples: int = 65,
) -> float:
    """
    Calculates the Mean Dose Error (MDE) for a given model.

    The MDE measures the mean difference between the true best dose and the dose selected by the model.
    (1/n) * \sum{(d^*_i - d-hat^*_i)^2}

    Parameters:
        x (np.ndarray): The covariates.
        t (np.ndarray): The treatments.
        response (Callable): A function representing the true response.
        model (Callable): The model to be evaluated.
        num_integration_samples (int, optional): The number of samples to be used for the integration. Defaults to 65.

    Returns:
        float: The Mean Dose Error of the model.
    """
    num_obs = x.shape[0]

    # Generate data
    x = np.repeat(x, repeats=num_integration_samples, axis=0)
    d = np.linspace(0, 1, num_integration_samples)
    d = np.tile(d, num_obs)
    t = np.repeat(t, repeats=num_integration_samples)

    # Get true outcomes
    y = response(x, d, t)
    # Get predictions
    y_hat = model.predict(x, d, t)

    # Get mean dose error
    squared_dose_errors = []
    y_chunks = y.reshape(-1, num_integration_samples)
    y_hat_chunks = y_hat.reshape(-1, num_integration_samples)
    d_chunks = d.reshape(-1, num_integration_samples)

    # Get errors
    for y_chunk, y_hat_chunk, d_chunk in zip(y_chunks, y_hat_chunks, d_chunks):
        pred_best_id = np.argmax(y_hat_chunk)
        actual_best_id = np.argmax(y_chunk)
        squared_dose_error = (d_chunk[pred_best_id] - d_chunk[actual_best_id]) ** 2
        squared_dose_errors.append(squared_dose_error)

    return np.sqrt(np.mean(squared_dose_errors))

