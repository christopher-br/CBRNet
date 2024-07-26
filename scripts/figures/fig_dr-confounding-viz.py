## KEY SETTINGS
#####################################

DIR = ".../CBRNet"
DATA_NAME = "drybean_1"

# Chg os and sys path
import os
import sys
os.chdir(DIR)
sys.path.append(DIR)

# Num of experiments per data configuration
NUM_ITERATIONS = 10


# LOAD MODULES
#####################################

# Standard library
import warnings

# Third party
from tqdm import tqdm

# Proprietary
from src.data.drybean_1 import load_data

from src.utils.viz import dose_plot, dr_plot


## SETUP
#####################################

# Disable device summaries
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.FileHandler("core.log"))

# Disable warnings
warnings.filterwarnings("ignore")


## CREATE PLOTS
#####################################

# Confounded

data = load_data(bias_inter=0.66, bias_intra=3)

# Generate dose plot
dose_plot(data.d_train, w=4, h=4, file_name="dry-bean-viz_conf_doses.pdf")

# Generate dr plot
dr_plot(data.x_train, data.d_train, data.t_train, data.ground_truth, w=4, h=4, file_name="dry-bean-viz_conf_drs.pdf")

# Unconfounded

data = load_data(bias_inter=0., bias_intra=3)

# Generate dose plot
dose_plot(data.d_train, w=4, h=4, file_name="dry-bean-viz_unconf_doses.pdf")

# Generate dr plot
dr_plot(data.x_train, data.d_train, data.t_train, data.ground_truth, w=4, h=4, file_name="dry-bean-viz_unconf_drs.pdf")