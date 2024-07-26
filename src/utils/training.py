# LOAD MODULES
# Standard library
from typing import Callable, Dict, Optional, Union
import random
import itertools

# Third party
import numpy as np
from tqdm import tqdm

# Proprietary
from src.data.utils import ContinuousData

def train_val_tuner(
    data: ContinuousData,
    model: Callable,
    parameters: dict,
    name: str = "method",
    num_combinations: Optional[int] = None,
):
    """
    Performs training and validation tuning on a given model with specified parameters.

    Parameters:
    data (ContinuousData or BinaryData): The dataset to be used for training and validation.
    model (Callable): The machine learning model to be tuned.
    parameters (dict): The parameters for the model.
    name (str, optional): The name of the method. Defaults to "method".
    num_combinations (int, optional): The number of parameter combinations to consider. If None, all combinations are considered. Defaults to None.

    Returns:
    final_model (Callable): The tuned model.
    best_parameters (dict): The best found settings for the model parameters.
    """
    # Seed
    random.seed(42)
    
    # Ini error
    current_best = np.inf
        
    # Save combinations and shuffle
    combinations = list(itertools.product(*parameters.values()))
    random.shuffle(combinations)
    
    # Sample combinations for random search
    if (num_combinations is not None) and (num_combinations < len(combinations)):
        combinations = combinations[:num_combinations]

    # Iterate over all combinations
    for combination in tqdm(combinations, leave=False, desc="Tune " + name):
        # Save settings of current iteration
        settings = dict(zip(parameters.keys(), combination))

        # Set up model
        estimator = model(**settings)

        # Fit model
        estimator.fit(data.x_train, data.y_train, data.d_train, data.t_train)

        # Score
        score = estimator.score(data.x_val, data.y_val, data.d_val, data.t_val)

        # Evaluate if better than current best
        if current_best > score:
            # Set best settings
            best_parameters = settings
            current_best = score

    # Train final model
    final_model = model(**best_parameters)
    final_model.fit(data.x_train, data.y_train, data.d_train, data.t_train)

    return final_model, best_parameters

def cv_tuner(
    data: ContinuousData,
    model: Callable,
    parameters: dict,
    name: str = "method",
    num_combinations: Optional[int] = None,
    num_folds: int = 5,
):
    """
    Performs cross-validation tuning on a given model with specified parameters.

    Parameters:
    data (ContinuousData or BinaryData): The dataset to be used for training and validation.
    model (Callable): The machine learning model to be tuned.
    parameters (dict): The parameters for the model.
    name (str, optional): The name of the method. Defaults to "method".
    num_combinations (int, optional): The number of parameter combinations to consider. If None, all combinations are considered. Defaults to None.
    num_folds (int, optional): The number of folds for cross-validation. Defaults to 5.

    Returns:
    final_model (Callable): The tuned model.
    best_settings (dict): The best found settings for the model parameters.
    """
    # Results array
    results = []
    
    # Seed
    random.seed(42)
    
    # Get train ids (all from train and val set)
    train_ids = data.train_val_ids
    
    # Get folds
    fold_ids = np.array_split(data.train_val_ids, num_folds)
    
    # Get parameter combinations
    combinations = list(itertools.product(*parameters.values()))
    random.shuffle(combinations)
    
    # Sample combinations for random search
    if (num_combinations is not None) and (num_combinations < len(combinations)):
        combinations = combinations[:num_combinations]
    
    # Iterate
    for combination in tqdm(combinations, leave=False, desc="Tune " + name):
        # Save settings
        settings = dict(zip(parameters.keys(), combination))
        
        # Fold results array
        fold_results = []
        for f in tqdm(range(num_folds), desc="Iterate over folds", leave=False):
            x_fold_train = data.x[fold_ids[f]]
            x_fold_val = data.x[np.setdiff1d(train_ids, fold_ids[f])]
            y_fold_train = data.y[fold_ids[f]]
            y_fold_val = data.y[np.setdiff1d(train_ids, fold_ids[f])]
            d_fold_train = data.d[fold_ids[f]]
            d_fold_val = data.d[np.setdiff1d(train_ids, fold_ids[f])]
            t_fold_train = data.t[fold_ids[f]]
            t_fold_val = data.t[np.setdiff1d(train_ids, fold_ids[f])]
            
            estimator = model(**settings)
            
            estimator.fit(x_fold_train, y_fold_train, d_fold_train, t_fold_train)
            fold_results.append(estimator.score(x_fold_val, y_fold_val, d_fold_val, t_fold_val))
        
        # Append mean score to overall results array
        results.append(np.mean(fold_results))
        
    best_option = np.argmin(results)
    best_settings = dict(zip(parameters.keys(), combinations[best_option]))
    
    final_model = model(**best_settings)
    final_model.fit(data.x[train_ids], data.y[train_ids], data.d[train_ids], data.t[train_ids])
    
    return final_model, best_settings
