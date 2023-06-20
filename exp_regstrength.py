# Global parameters
n_iter = 10
d_bias = 0.5
s_bias = 5
iter_strengths = [0.0, 0.5, 5.0, 50.0, 100.0, 500.0]
accelerator = 'cpu'
num_workers = 1
verbose = False
save_logs = False

# Dataset parameters
data_path = 'datasets/tcga.csv'
response_type = 'quadratic'
binary_targets = False

# Set ws
import os
os.chdir('...')

# Import modules
import numpy as np
import wandb
import logging
import warnings
import gc
import copy
import torch

from tqdm import tqdm

from modules.MLP import MLP
from modules.CBRNets import CBRNet
from modules.DataGen import gen_dataset
from utils.Wandb import getAPI
from utils.Print import summary_plot
from utils.Tune import tuner

# Filter workers warning in pytroch
warnings.filterwarnings('ignore', '.*does not have many workers.*')
warnings.filterwarnings('ignore', '.*overflow encountered.*')
warnings.filterwarnings('ignore', '.*MPS available.*')

# Set False if full log to be printed
if (save_logs == False):
    # configure logging at the root level of Lightning
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger('pytorch_lightning.core')
    logger.addHandler(logging.FileHandler('core.log'))

# Disable automatic gc
gc.disable()

# Ini step id
step_id = 0

# Log in
wandb.login(key=getAPI())
wandb.init(project='...', entity='...', reinit=False, save_code=True)

# Start iterating
for n in tqdm(range(n_iter), leave=False, desc='Iteration'):
    # Log step details
    wandb.log({'Distr. bias': d_bias,
                'Selec. bias': s_bias,
                'Iteration': n},
                step=step_id)
    
    ######################
    # Load data
    ######################
    
    # Load np dataset
    x_data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    
    # Ini dataset
    dataset_params = dict()
    dataset_params['dosageSelectionBias'] = s_bias
    dataset_params['dosageDistributionBias'] = d_bias
    dataset_params['responseType'] = response_type
    dataset_params['binaryTargets'] = binary_targets
    dataset_params['testFraction'] = 0.2
    dataset_params['valFraction'] = 0.1
    dataset_params['noise'] = 1.
    dataset_params['numObs'] = x_data.shape[0]
    dataset_params['numVars'] = x_data.shape[1]
    dataset_params['seed'] = 5*n
    
    # Get number of variables
    num_features = x_data.shape[1]
    
    # Generate dataset
    dataset, dataset_train, dataset_val, dataset_test = gen_dataset(dataset_params, x=x_data)
    
    ######################
    # Train models
    ######################

    #######
    # MLP #
    #######
            
    # Params for tuning
    params = {
        # System parameters
        'verbose': [verbose],
        'accelerator': [accelerator],
        # Model architecture
        'inputSize': [num_features],
        'numLayers': [2],
        'hiddenSize': [32, 48],
        'activationFct': ['elu'], # Can be 'relu', 'elu', 'leaky_relu', 'sigmoid', 'linear'
        # Training parameters
        'numSteps': [5000],
        'learningRate': [0.01, 0.025],
        'batchSize': [64]
    }
    
    # Tune
    model = tuner(MLP,
                    dataset_train,
                    dataset_val,
                    params,
                    'MLP')
    
    # Evaluate
    mise, pe, fe = model.computeMetrics(dataset_test)
    
    del model
    
    # Log results
    wandb.log({'MISE MLP': mise,
               'PE MLP': pe,
               'FE MLP': fe},
              step=step_id)
    
    
    ##########
    # CBRNet #
    ##########
    
    # Iterate over linear mmd
    for strength in tqdm(iter_strengths, desc='Iterate over lin MMD', leave=False):
        # Params for tuning
        params = {
            # System parameters
            'verbose': [verbose],
            'accelerator': [accelerator],
            # Model architecture
            'inputSize': [num_features],
            'numRepresentationLayers': [2],
            'numInferenceLayers': [2],
            'numBins': [5],
            'hiddenSize': [32, 48],
            'mmdType': ['linear'], # Can be 'linear', 'rbf', 'none'
            'binningMethod': ['kmeans'], # Can be 'linear', 'quantile', 'jenks', 'kde', 'kmeans'
            'regularizeBeta': [strength],
            'activationFct': ['elu'], # Can be 'relu', 'elu', 'leaky_relu', 'sigmoid', 'linear'
            # Training parameters
            'numSteps': [5000],
            'learningRate': [0.01, 0.025],
            'batchSize': [64]
        }
        
        # Tune
        model = tuner(CBRNet,
                      dataset_train,
                      dataset_val,
                      params,
                      'CBRNet')
        
        # Evaluate
        mise, pe, fe = model.computeMetrics(dataset_test)
        
        del model
        
        # Set names for saving results
        mise_name = str('MISE l:') + str(round(strength, 2))
        pe_name = str('PE l:') + str(round(strength, 2))
        fe_name = str('FE l:') + str(round(strength, 2))
                    
        # Log results
        wandb.log({mise_name: mise,
                   pe_name: pe,
                   fe_name: fe},
                  step=step_id)
        
        # Run gc
        gc.collect()
        
    # Iterate over rbf mmd
    for strength in tqdm(iter_strengths, desc='Iterate over lin MMD', leave=False):
        # Params for tuning
        params = {
            # System parameters
            'verbose': [verbose],
            'accelerator': [accelerator],
            # Model architecture
            'inputSize': [num_features],
            'numRepresentationLayers': [2],
            'numInferenceLayers': [2],
            'numBins': [3],
            'hiddenSize': [32, 48],
            'mmdType': ['rbf'], # Can be 'linear', 'rbf', 'none'
            'binningMethod': ['kmeans'], # Can be 'linear', 'quantile', 'jenks', 'kde', 'kmeans'
            'regularizeBeta': [strength],
            'activationFct': ['elu'], # Can be 'relu', 'elu', 'leaky_relu', 'sigmoid', 'linear'
            # Training parameters
            'numSteps': [5000],
            'learningRate': [0.01, 0.025],
            'batchSize': [64]
        }
        
        # Tune
        model = tuner(CBRNet,
                      dataset_train,
                      dataset_val,
                      params,
                      'CBRNet rbf')
        
        # Evaluate
        mise, pe, fe = model.computeMetrics(dataset_test)
        
        del model
        
        # Set names for saving results
        mise_name = str('MISE rbf l:') + str(round(strength, 2))
        pe_name = str('PE rbf l:') + str(round(strength, 2))
        fe_name = str('FE rbf l:') + str(round(strength, 2))
                    
        # Log results
        wandb.log({mise_name: mise,
                   pe_name: pe,
                   fe_name: fe},
                  step=step_id)
        
        # Run gc
        gc.collect()

    ######################
    # Finish iteration
    ######################
    
    # Run gc
    gc.collect()
    
    # Increment step id
    step_id = step_id + 1

