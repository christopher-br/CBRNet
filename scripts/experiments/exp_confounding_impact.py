# KEY SETTINGS
#####################################

DIR = ".../CBRNet"
DATA_NAME = "drybean_1"
EXP_NAME = "exp_confounding_impact"

# Chg os and sys path
import os
import sys
os.chdir(DIR)
sys.path.append(DIR)

# Num of experiments per data configuration
NUM_ITERATIONS = 10

# Number of parameter combinations to consider
RANDOM_SEARCH_N = 5


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
from src.data.drybean_1 import load_data
from src.methods.neural import MLP
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
DATA_PARAS = load_config("config/experiments/config.yaml")[EXP_NAME]

# Add iteration indicator to data paras
DATA_PARAS["keys"].append("seed")
DATA_PARAS["keys"] = tuple(DATA_PARAS["keys"])
new_tuples = []
for i in range(NUM_ITERATIONS):
    for tup in DATA_PARAS["tuples"]:
        new_tuples.append(tuple(tup + [i]))
DATA_PARAS["tuples"] = new_tuples

# Generate tracker
check_create_csv(EXP_NAME+"_tracker.csv", DATA_PARAS["keys"])


## CUSTOM FUNCTIONS
#####################################

def update_dict(dict, data, model, name):
    mise = mise_metric(data.x_test, data.t_test, data.ground_truth, model)
    dict.update({f"MISE {name}": mise})


## ITERATE OVER DATA COMBINATIONS
#####################################

for comb in tqdm(DATA_PARAS["tuples"], desc="Iterate over data combinations", leave=False):
    completed = get_rows(EXP_NAME+"_tracker.csv")
    if comb in completed:
        continue
    else:
        add_row(
            row=comb,
            file_path=EXP_NAME+"_tracker.csv",
        )
        
    data_settings = dict(zip(DATA_PARAS["keys"], comb))
    
    results = {}
    results.update(data_settings)
    
    data = load_data(**data_settings)
    
    # TRAIN MODELS
    
    # mlp
    name = "mlp"
    model = MLP(
        input_size=data.x.shape[1],
        learning_rate=0.01,
        batch_size=128,
        num_steps=2500,
    )
    model.fit(data.x_train, data.y_train, data.d_train, data.t_train)
    
    update_dict(results, data, model, name)

    # FINALIZE ITERATION
    add_dict("res/"+EXP_NAME+".csv", results)