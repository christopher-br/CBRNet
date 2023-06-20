# Written by Christopher Bockel-Rickermann. Copyright (c) 2023

############
# Preamble #
############

# Directory
directory = '...'

# Dataset in directory
data_path = 'datasets/tcga.csv'

# Ground truth
response_type = 'quadratic'

# Number of iterations
n_iter = 10

# Biases to iterate over
# Distribution biases [0, inf)
distribution_biases = [0,5,10,50]

# Selection biases [0, 1)
selection_biases = [0, 0.25, 0.5, 0.75, 0.99]

# Verbosity of models
verbose = False

# Save logs locally
save_logs = False

################
# Load modules #
################

# Full modules from pip
import os
import numpy as np
import wandb
import logging
import warnings
import gc
import copy
import torch

# Fcts from modules from pip
from tqdm import tqdm

# Custom models
from modules.MLP import MLP
from modules.DRNets import DRNet
from modules.CBRNets import CBRNet
from modules.GPS import GPS
from modules.LinReg import LinReg
from modules.VCNets import VCNet

# Custome functions
from modules.DataGen import gen_dataset
from utils.Wandb import getAPI
from utils.Print import summary_plot
from utils.Tune import tuner

############
# Settings #
############

# Compute parameters
accelerator = 'cpu'
num_workers = 1

# Set ws
os.chdir('...')

# Filter workers warning in pytroch
warnings.filterwarnings('ignore', '.*does not have many workers.*')
warnings.filterwarnings('ignore', '.*overflow encountered.*')
warnings.filterwarnings('ignore', '.*MPS available.*')
warnings.filterwarnings("ignore", ".*on tensors of dimension other than 2 to reverse their shape is deprecated.*")
warnings.filterwarnings("ignore", ".*lbfgs failed to converge.+")

# Define logging behavior
if (save_logs == False):
    # configure logging at the root level of Lightning
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger('pytorch_lightning.core')
    logger.addHandler(logging.FileHandler('core.log'))
    
# Disable automatic gc
gc.disable()

# Set counter for logging
step_id = 0

# Log into WandB
wandb.login(key=getAPI())
wandb.init(project='...', entity='...', reinit=False, save_code=True)

##################
# Run experiment #
##################

# Start iterating
for d_bias in tqdm(distribution_biases, desc='Iterating over distr. biases'):
    for s_bias in tqdm(selection_biases, desc='Iterating over selec. biases'):
        for n in tqdm(range(n_iter), leave=False, desc='Iteration'):
            
            ########################
            # Initilaize iteration #
            ########################
            
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
            
            # Plot dosages
            dosages = dataset['d']
            clusters = dataset['c']
            
            # Get an average observation
            obs = torch.Tensor(torch.mean(torch.Tensor(dataset['x']), axis=0))
            
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
            
            # Get DR
            true_dr = model.getTrueDR(obs)
            mlp_dr = model.getDR(obs)
            mise_mlp = mise
            
            del model
            
            # Log results
            wandb.log({'MISE MLP': mise,
                       'PE MLP': pe,
                       'FE MLP': fe},
                      step=step_id)
            
            gc.collect()
            
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
                'numBins': [10],
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
                          'DRNet')
            
            # Evaluate
            mise, pe, fe = model.computeMetrics(dataset_test)
            
            # Get DR
            drnet_dr = model.getDR(obs)
            mise_drnet = mise
            
            del model
            
            # Log results
            wandb.log({'MISE DRNet': mise,
                       'PE DRNet': pe,
                       'FE DRNet': fe},
                      step=step_id)
            
            gc.collect()
            
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
            
            # Get DR
            cbrnet_dr = model.getDR(obs)
            mise_cbrnet = mise
            
            del model
            
            # Log results
            wandb.log({'MISE CBRNet': mise,
                       'PE CBRNet': pe,
                       'FE CBRNet': fe},
                      step=step_id)
            
            gc.collect()
            
            #######
            # GPS #
            #######
            
            # Params for tuning
            params = {
                'treatPolyDegree': [1],
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
            
            # Get DR
            gps_dr = model.getDR(obs)
            mise_gps = mise
            
            del model
            
            # Log results
            wandb.log({'MISE GPS': mise,
                       'PE GPS': pe,
                       'FE GPS': fe},
                      step=step_id)
            
            gc.collect()
            
            ##########
            # LinReg #
            ##########
            
            # Params for tuning
            params = {
                'polyDegree': [2],
                'penalty' : [False]
            }
            
            # Tune
            model = tuner(LinReg,
                          dataset_train,
                          dataset_val,
                          params,
                          'Linear Regression')
            
            # Evaluate
            mise, pe, fe = model.computeMetrics(dataset_test)
            
            # Get DR
            linreg_dr = model.getDR(obs)
            mise_linreg = mise
            
            del model
            
            # Log results
            wandb.log({'MISE LinReg': mise,
                       'PE LinReg': pe,
                       'FE LinReg': fe},
                      step=step_id)
            
            gc.collect()
            
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
            
            # Get DR
            vcnet_dr = model.getDR(obs)
            mise_vcnet = mise
            
            del model
            
            # Log results
            wandb.log({'MISE VCNet': mise,
                    'PE VCNet': pe,
                    'FE VCNet': fe},
                    step=step_id)
            
            gc.collect()
            
            ######################
            # Finish iteration
            ######################
            
            # Log plot
            mlp_dr = np.array(mlp_dr)
            drnet_dr = np.array(drnet_dr)
            cbrnet_dr = np.array(cbrnet_dr)
            linreg_dr = np.array(linreg_dr)
            gps_dr = np.array(gps_dr)
            vcnet_dr = np.array(vcnet_dr)
            figure = summary_plot(true_dr, 
                                  mlp_dr, drnet_dr, cbrnet_dr, linreg_dr, gps_dr, vcnet_dr, 
                                  mise_mlp, mise_drnet, mise_cbrnet, mise_linreg, mise_gps, mise_vcnet, 
                                  dosages, clusters, s_bias, d_bias)
            wandb.log({'Dose Response': figure},
                      step=step_id)
            
            # Run gc
            gc.collect()
            
            # Increment step id
            step_id = step_id + 1
        