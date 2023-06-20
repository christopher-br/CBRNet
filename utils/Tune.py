# Written by Christopher Bockel-Rickermann. Copyright (c) 2023

import numpy as np
import itertools
import copy

from tqdm import tqdm

def tuner(model, dataset_train, dataset_val, dictionary, name='Model'):
    # Set val error to infinity
    val_error = np.inf
    
    # Save param names
    param_names = list(dictionary.keys())
    
    # Save values
    param_values = list(dictionary.values())
    
    # Iterate over parameters
    for combination in tqdm(itertools.product(*param_values), leave=False, desc=('Tune '+name)):
        # Save parameters
        parameters = dict(zip(param_names, combination))
        
        # Set up model
        temp = model(parameters, dataset_train)
        
        # Get validation error
        temp_error = temp.validateModel(dataset_val)
        
        # Save new lowest error and save model
        if (temp_error < val_error):
            val_error = temp_error
            best_config = copy.deepcopy(temp)
        
        # Delete model
        del temp
        
    # Return best model
    return best_config