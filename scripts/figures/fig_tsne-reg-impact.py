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

reg_strength = 0.1


# LOAD MODULES
#####################################

# Standard library
import warnings

# Third party
from tqdm import tqdm

# Proprietary
from src.data.drybean_1 import load_data
from src.methods.neural import CBRNet
from src.methods.utils.classes import TorchDataset
from src.methods.utils.cbrnet_utils import MMD, Wasserstein

from src.utils.viz import tsne_plot, dose_plot


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

data = load_data(bias_inter=0.66, bias_intra=3)
# Save as torchDataset
t_data_test = TorchDataset(data.x_test, data.y_test, data.d_test, data.t_test)

# Plot input space
tsne_plot(data.x_test, data.d_test, file_name="fig:tsne-reg-impact_input.pdf")

# TRAIN MODELS
# CBRNet wo reg
model = CBRNet(
    input_size=data.x.shape[1],
    learning_rate=0.01,
    batch_size=128,
    num_steps=2500,
    IPM=MMD('rbf'),
    regularization_ipm=0.0,
    hidden_size=32,
)
model.fit(data.x_train, data.y_train, data.d_train, data.t_train)

_, rep = model.forward(t_data_test.x, t_data_test.d, t_data_test.t)
rep = rep.detach().numpy()

tsne_plot(rep, data.d_test, file_name="fig:tsne-reg-impact_no-reg.pdf")

# CBRNet w reg RBF
model = CBRNet(
    input_size=data.x.shape[1],
    learning_rate=0.01,
    batch_size=128,
    num_steps=2500,
    IPM=MMD('rbf'),
    regularization_ipm=reg_strength,
    hidden_size=32,
)
model.fit(data.x_train, data.y_train, data.d_train, data.t_train)

_, rep = model.forward(t_data_test.x, t_data_test.d, t_data_test.t)
rep = rep.detach().numpy()

tsne_plot(rep, data.d_test, file_name="fig:tsne-reg-impact_reg_rbf.pdf")

# Create dose histogram
dose_plot(data.d_test, w=4, h = 4, file_name="fig:tsne-reg-impact_doses.pdf", num_bins=25)

# CBRNet w reg LIN
model = CBRNet(
    input_size=data.x.shape[1],
    learning_rate=0.01,
    batch_size=128,
    num_steps=2500,
    IPM=MMD('linear'),
    regularization_ipm=reg_strength,
    hidden_size=32,
)
model.fit(data.x_train, data.y_train, data.d_train, data.t_train)

_, rep = model.forward(t_data_test.x, t_data_test.d, t_data_test.t)
rep = rep.detach().numpy()

tsne_plot(rep, data.d_test, file_name="fig:tsne-reg-impact_reg_lin.pdf")

# Create dose histogram
dose_plot(data.d_test, w=4, h = 4, file_name="fig:tsne-reg-impact_doses.pdf", num_bins=25)

# CBRNet w reg WAS
model = CBRNet(
    input_size=data.x.shape[1],
    learning_rate=0.01,
    batch_size=128,
    num_steps=2500,
    IPM=Wasserstein(),
    regularization_ipm=reg_strength,
    hidden_size=32,
)
model.fit(data.x_train, data.y_train, data.d_train, data.t_train)

_, rep = model.forward(t_data_test.x, t_data_test.d, t_data_test.t)
rep = rep.detach().numpy()

tsne_plot(rep, data.d_test, file_name="fig:tsne-reg-impact_reg_was.pdf")

# Create dose histogram
dose_plot(data.d_test, w=4, h = 4, file_name="fig:tsne-reg-impact_doses.pdf", num_bins=25)