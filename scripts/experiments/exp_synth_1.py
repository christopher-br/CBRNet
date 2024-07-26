## KEY SETTINGS
#####################################

DIR = ".../CBRNet"
EXP_NAME = "exp_synth_1"

# Chg os and sys path
import os
import sys
os.chdir(DIR)
sys.path.append(DIR)

# Num of experiments per data configuration
NUM_ITERATIONS = 10

# Number of parameter combinations to consider
RANDOM_SEARCH_N = 10


# LOAD MODULES
#####################################

# Standard library
import warnings

# Third party
from tqdm import tqdm

# Regressors:
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Proprietary
from src.data.synth_1 import load_data
from src.methods.other import SLearner
from src.methods.neural import DRNet, CBRNet, MLP, VCNet
from src.methods.utils.cbrnet_utils import MMD, Wasserstein
from src.methods.utils.regressors import LinearRegression
from src.utils.metrics import mise_metric
from src.utils.setup import (
    load_config,
    check_create_csv,
    get_rows,
    add_row,
    add_dict,
)
from src.utils.training import train_val_tuner


## SETUP
#####################################

# Disable device summaries
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.FileHandler("core.log"))

# Disable warnings
warnings.filterwarnings("ignore")

# Load config
HYPERPARAS = load_config("config/methods/config.yaml")


## CUSTOM FUNCTIONS
#####################################

def update_dict(dict, data, model, name):
    mise = mise_metric(data.x_test, data.t_test, data.ground_truth, model)
    dict.update({f"MISE {name}": mise})


## ITERATE OVER DATA COMBINATIONS
#####################################

for n in tqdm(range(NUM_ITERATIONS), desc="Iterate", leave=False):
    results = {}
    results.update({"seed": n})
    
    data = load_data(seed=n)
    
    # TRAIN MODELS
    
    # mlp
    name = "mlp"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    model, best_params = train_val_tuner(
        data = data,
        model = MLP,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)

    # DRNet
    name = "drnet"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    model, best_params = train_val_tuner(
        data = data,
        model = DRNet,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)
    
    # VCNet
    name = "vcnet"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    model, best_params = train_val_tuner(
        data = data,
        model = VCNet,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    update_dict(results, data, model, name)
    
    # CBRNet lin
    name = "cbrnet"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    parameters.update({"IPM": [MMD("linear")]})
    model, best_params = train_val_tuner(
        data = data,
        model = CBRNet,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    name = "cbrnet-lin"
    update_dict(results, data, model, name)
    
    # CBRNet rbf
    name = "cbrnet"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    parameters.update({"IPM": [MMD("rbf")]})
    model, best_params = train_val_tuner(
        data = data,
        model = CBRNet,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    name = "cbrnet-rbf"
    update_dict(results, data, model, name)
    
    # CBRNet lin
    name = "cbrnet"
    parameters = HYPERPARAS[name]
    parameters.update({"input_size": [data.x.shape[1]]})
    parameters.update({"IPM": [Wasserstein()]})
    model, best_params = train_val_tuner(
        data = data,
        model = CBRNet,
        parameters = parameters,
        name = name,
        num_combinations = RANDOM_SEARCH_N,
    )
    
    name = "cbrnet-was"
    update_dict(results, data, model, name)
    
    # FINALIZE ITERATION
    add_dict("res/"+EXP_NAME+".csv", results)