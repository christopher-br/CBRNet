# Global parameters
n_iter = 10
accelerator = 'cpu'
num_workers = 1
verbose = False
save_logs = False

# Dataset parameters
data_path = 'datasets/tcga.csv' # Use 'crdt_gmsc', 'tcga', 'tcga_pca'
binary_targets = False

# Set ws
import os
os.chdir('...')

# Network parameters
# Activation fcts.: 'relu', 'elu', 'leaky_relu', 'sigmoid', 'linear'
# Binning methods: 'linear', 'quantile', 'kde', 'jenks', 'kmeans'
# MMD types: 'linear', 'rbf', or 'none' > Specified in each loop!

# Import modules
import numpy as np
import wandb
import logging
import warnings
import gc
import copy
import torch

from tqdm import tqdm

from modules.DRNets import DRNet
from modules.MLP import MLP
from modules.CBRNets import CBRNet
from modules.GPS import GPS
from modules.VCNets import VCNet
from modules.DataGen import gen_dataset
from utils.Wandb import getAPI
from utils.Print import summary_plot
from utils.Tune import tuner

from benchmarks.Bica import gen_bica_dataset, torchDataset

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
    wandb.log({'Iteration': n},
                step=step_id)
    
    ######################
    # Bica (2020)
    ######################
    
    # Load np dataset
    x_data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    
    # Iterate over different response types
    for r_type in ['quadratic','sine','linear']:
        # Set dataset params
        dataset_params = dict()
        dataset_params['bias'] = 5
        dataset_params['responseType'] = r_type
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
        
        dataset, dataset_train, dataset_val, dataset_test = gen_bica_dataset(dataset_params, x=x_data)
        
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
        
        # Del previous models
        del model
                    
        # Log results
        mise_name = 'Bica ' + str(r_type) + ' MISE MLP'
        pe_name = 'Bica ' + str(r_type) + ' PE MLP'
        fe_name = 'Bica ' + str(r_type) + ' FE MLP'
        wandb.log({mise_name: mise,
                   pe_name: pe,
                   fe_name: fe},
                  step=step_id)
        
        #########
        # DRNet #
        #########
        
        # Params for tuning
        params = {
            # System parameters
            'verbose': [verbose],
            'accelerator': [accelerator],
            # Model architecture
            'inputSize': [num_features],
            'numLayers': [2],
            'numBins': [5],
            'hiddenSize': [32, 48],
            'activationFct': ['elu'], # Can be 'relu', 'elu', 'leaky_relu', 'sigmoid', 'linear'
            # Training parameters
            'numSteps': [5000],
            'learningRate': [0.01, 0.025],
            'batchSize': [64]
        }
        
        # Tune
        model = tuner(DRNet,
                      dataset_train,
                      dataset_val,
                      params,
                      'MLP')
        
        # Evaluate
        mise, pe, fe = model.computeMetrics(dataset_test)
        
        # Del previous models
        del model
                    
        # Log results
        mise_name = 'Bica ' + str(r_type) + ' MISE DRNet'
        pe_name = 'Bica ' + str(r_type) + ' PE DRNet'
        fe_name = 'Bica ' + str(r_type) + ' FE DRNet'
        wandb.log({mise_name: mise,
                   pe_name: pe,
                   fe_name: fe},
                  step=step_id)
        
        ##########
        # CBRNet #
        ##########
        
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
            'mmdType': ['rbf'], # Can be 'linear', 'rbf', 'none'
            'binningMethod': ['kmeans'], # Can be 'linear', 'quantile', 'jenks', 'kde', 'kmeans'
            'regularizeBeta': [0.0, 0.5, 5.0, 50.0],
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
        
        # Del previous models
        del model
                    
        # Log results
        mise_name = 'Bica ' + str(r_type) + ' MISE CBRNet'
        pe_name = 'Bica ' + str(r_type) + ' PE CBRNet'
        fe_name = 'Bica ' + str(r_type) + ' FE CBRNet'
        wandb.log({mise_name: mise,
                   pe_name: pe,
                   fe_name: fe},
                  step=step_id)
        
        #######
        # GPS #
        #######
        
        # Params for tuning
        params = {
            'treatPolyDegree': [2],
            'outcomePolyDegree' : [2]
        }
        
        # Tune
        model = tuner(GPS,
                      dataset_train,
                      dataset_val,
                      params,
                      'GPS')
        
        # Evaluate
        mise, pe, fe = model.computeMetrics(dataset_test)
        
        # Del previous models
        del model
                    
        # Log results
        mise_name = 'Bica ' + str(r_type) + ' MISE GPS'
        pe_name = 'Bica ' + str(r_type) + ' PE GPS'
        fe_name = 'Bica ' + str(r_type) + ' FE GPS'
        wandb.log({mise_name: mise,
                   pe_name: pe,
                   fe_name: fe},
                  step=step_id)
        
        #########
        # VCNet #
        #########
        
        # Params for tuning
        params = {
            'learningRate': [0.01],
            'batchSize': [500],
            'hiddenSize': [50],
            'numEpochs': [400],
            'numGrid': [10],
            'knots': [[0.33, 0.66]],
            'degree': [2],
            'targetReg': [True],
            'inputSize': [num_features]
        }
        
        # Tune
        model = tuner(VCNet,
                      dataset_train,
                      dataset_val,
                      params,
                      'VCNet')
        
        # Evaluate
        mise, pe, fe = model.computeMetrics(dataset_test)
        
        # Del previous models
        del model
                    
        # Log results
        mise_name = 'Bica ' + str(r_type) + ' MISE VCNet'
        pe_name = 'Bica ' + str(r_type) + ' PE VCNet'
        fe_name = 'Bica ' + str(r_type) + ' FE VCNet'
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
