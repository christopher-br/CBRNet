# Load necessary modules
import numpy as np
from tqdm import tqdm

from scipy.integrate import romb
from sklearn.neighbors import BallTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from utils.Responses import get_outcome
from utils.Loss import mmd_lin, mmd_rbf, mmd_none
from utils.Binning import lin_bins, quantile_bins, jenks_bins, kde_bins, kmeans_bins

from modules.DataGen import torchDataset

class DRNet(pl.LightningModule):
    """
    The DRNet model.
    
    Attributes:
        config (dict): A dictionary with all parameters to create and train the model
        dataset (dict): Dataset for training
        
    Config entries:
        'learningRate' (float): The learning rate
        'batchSize' (int): The batch size
        'numSteps' (int): The number of training steps
        'numLayers' (int): The number of hidden and head layers
        'numBins' (int): The number of individual bins
        'inputSize' (int): The input size
        'hiddenSize' (int): The size of each hidden layer 
        'activationFct' (str): The activation function to use in the forward pass
        'verbose' (bool): Verbosity during network training
    """
    
    def __init__(self, config, dataset):
        """
        The constructor of the DRNet class.
        
        Parameters:
            config (dict): A dictionary with all parameters to create and train the model
            dataset (dict): Dataset for training
            
        Config entries:
            'learningRate' (float): The learning rate
            'batchSize' (int): The batch size
            'numSteps' (int): The number of training steps
            'numLayers' (int): The number of hidden and head layers
            'numBins' (int): The number of individual bins
            'inputSize' (int): The input size
            'hiddenSize' (int): The size of each hidden layer 
            'activationFct' (str): The activation function to use in the forward pass
            'verbose' (bool): Verbosity during network training
        """
        # Ini the super module
        super(DRNet, self).__init__()
        
        torch.manual_seed(42)
        
        # Save config
        self.config = config
        
        # Save training data
        self.trainDataset = dataset
        self.t_trainDataset = torchDataset(dataset)
        
        # Save settings
        self.learningRate = config.get('learningRate')
        self.batch_size = config.get('batchSize')
        self.num_steps = config.get('numSteps')
        self.num_layers = config.get('numLayers')
        self.num_bins = config.get('numBins')
        self.input_size = config.get('inputSize')
        self.hidden_size = config.get('hiddenSize')
        self.activation = config.get('activationFct')
        self.verbose = config.get('verbose')
        self.accelerator = config.get('accelerator')
        
        # Save activation functions in dict
        self.activation_map = {'linear': nn.Identity(),
                               'elu': nn.ELU(),
                               'relu': nn.ReLU(),
                               'leaky relu': nn.LeakyReLU(),
                               'sigmoid': nn.Sigmoid()}
        
        # Initialize trainer
        self.trainer = Trainer(max_steps=self.num_steps,
                               max_epochs=9999, # To not interupt step-wise iteration
                               fast_dev_run=False,
                               reload_dataloaders_every_n_epochs=False,
                               enable_progress_bar=self.verbose,
                               enable_checkpointing=False,
                               enable_model_summary=self.verbose,
                               logger=False,
                               accelerator=self.accelerator,
                               devices=1)
        
        # Define weights and bins
        # Save bins
        obs_binned, binner, n_bins = lin_bins(d=self.t_trainDataset.d, x=self.t_trainDataset.x, n_bins=self.num_bins)
        # Save binner fct to self
        self.binner = binner
        # Overwrite num_bins (only makes change for binning fct that calculated optiomal number of bins)
        self.num_bins = n_bins
        
        # Set number of heads to 1, if multihead is turned off, otherwise to num bins
        self.num_heads = self.num_bins
        
        # Structure
        # Shared layers
        # Initialize shared layers
        self.shared_layers = nn.Sequential(nn.Linear(self.input_size, self.hidden_size))
        self.shared_layers.append(self.activation_map[self.activation])
        # Add additional layers
        for i in range(self.num_layers - 1):
            self.shared_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.shared_layers.append(self.activation_map[self.activation])
        
        # Hidden head layers
        self.head_layers = nn.ModuleList()
        for i in range(self.num_heads):
            # Ini head network
            help_head = nn.ModuleList([nn.Linear(self.hidden_size + 1, self.hidden_size)])
            # Append layers
            for i in range(self.num_layers - 1):
                help_head.extend([nn.Linear(self.hidden_size + 1, self.hidden_size)])
            # Append output layer
            help_head.extend([nn.Linear(self.hidden_size, 1)])
            # Append to module list
            self.head_layers.append(help_head)
            
        # Train
        self.trainModel(self.trainDataset)
        
                       
    # Define forward step
    def forward(self, x, d):
        """
        The forward function of a DRNets model.
        
        Parameters:
            x (torch tensor): A torch tensor (usually passed over by a Dataloader)
            d (torch tensor): A torch tensor (usually passed over by a Dataloader)
        """
        # Feed through shared layers
        x = self.shared_layers(x)
        
        # Reshape d
        d = d.reshape(-1,1)
        
        # Add d to x
        x = torch.cat((x, d), dim=1)
        
        # Results array
        res = torch.zeros(x.shape[0], self.num_heads)
        
        # Iterate over heads
        for i in range(self.num_heads):
            # Pass through first layer
            helper_x = self.activation_map[self.activation](self.head_layers[i][0](x))
            # Iterate over remaining
            for j in range(1, self.num_layers):
                # Concat dosage
                helper_x = torch.cat((helper_x, d), dim=1)
                # Pass through and activate
                helper_x = self.activation_map[self.activation](self.head_layers[i][j](helper_x))
            # Calculate output
            helper_x = self.head_layers[i][self.num_layers](helper_x)
            
            # Attach to res
            res[:, i] = helper_x.reshape(-1)
                
        # Return
        return res
    
    # Training step
    def training_step(self, batch, batch_idx):
        """
        The training step function of a DRNets model.
        
        Parameters:
            batch (triple of torch tensors): A triple of torch tensors (usually passed over by a Dataloader)
        
        batch items:
            x (torch tensor)
            y (torch tensor)
            d (torch tensor)
        """
        # Get batch items
        x, y_true, d = batch
        
        # Get results of forward pass
        y = self(x, d)
        
        # Save copy
        y_help = y.detach().clone()
        
        # Get bins
        obs_binned = self.binner(d=d, x=x)
        
        # Assign true value to only the head corresponding to d
        for obs in range(x.shape[0]):
            y_help[obs, obs_binned[obs]] = y_true[obs]
        
        # Get the loss
        loss_mse = F.mse_loss(y, y_help) * self.num_heads
        
        return loss_mse
    
    def configure_optimizers(self):
        """
        The optimizer of a DRNets model
        """
        return torch.optim.Adam(self.parameters(), lr=self.learningRate)
    
    def dataloader(self, torchDataset, shuffle=True):
        """
        Generates a DataLoader for a DRNets model
        
        Parameters:
            torchDataset (torchDataset): A torchDataset
            shuffle (bool): An indicator if data should be shuffled everytime Dataloader is called
        """
        # Generate DataLoader
        return DataLoader(torchDataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def trainModel(self, dataset):
        """
        Fits a DRNets model to a dataset
        
        Parameters:
            dataset (dict): A dataset
        """
        # Safe data as torchDataset
        tData = torchDataset(dataset)
        # Define loader
        loader = self.dataloader(tData)
        # Fit
        self.trainer.fit(self, loader) 
        
    def validateModel(self, dataset):
        """
        Validates a DRNets model on a dataset, by calculating the mean error on the observations in the dataset
        
        Parameters:
            dataset (dict): A dataset
        """
        # Safe data as torchDataset
        tData = torchDataset(dataset)
        # Get true outcomes
        x, y_true, d = tData.get_data()
        
        # Generate predictions
        y = torch.Tensor(self.predictObservation(x, d).squeeze())
        
        # Get mse
        val_mse = F.mse_loss(y, y_true).detach().item()
        
        # Return mse
        return val_mse
    
    def predictObservation(self, x, d):
        """
        Generates predictions of a DRNets model for data
        
        Parameters:
            x (torch tensor): A torch tensor with observations
            d (torch tensor): A torch tensor with dosage information
        """
        # Get outcomes (array form)
        outcomes = self(x, d)
        
        # Get bins
        obs_binned = self.binner(d=d, x=x)
        
        # Gather predictions via bin id
        predictions = torch.gather(outcomes, 1, obs_binned.reshape(-1, 1))
            
        predictions = predictions.detach().numpy()
            
        return predictions
    
    def getDR(self, observation):
        """
        Generates the predicted dose response of a DRNets model for a single observation
        
        Parameters:
            observation (torch tensor): A torch tensor with observations
        """
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = torch.linspace(np.finfo(float).eps, 1, num_integration_samples).reshape(-1,1)
        
        x = observation.repeat(num_integration_samples, 1)
        
        dr_curve = self.predictObservation(x, treatment_strengths).squeeze()
        
        return dr_curve
    
    def getTrueDR(self, observation):
        """
        Generates the true dose response of a DRNets model for a single observation
        
        Parameters:
            observation (torch tensor): A torch tensor with observations
        """
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = torch.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Get the dr curve
        dr_curve = np.array([get_outcome(observation, self.trainDataset['v'], d, self.trainDataset['response']).detach().item() for d in treatment_strengths])
        
        return dr_curve
    
    def computeMetrics(self, dataset):
        """
        Generates MISE and PE values for a DRNets model on a torchDataset
        
        Parameters:
            dataset (dict): A dataset
        """
        tData = torchDataset(dataset)
        # Initialize result arrays
        mises = []
        pes = []
        fes = []
        
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 1. / num_integration_samples
        treatment_strengths = torch.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Get number of observations
        num_test_obs = dataset['x'].shape[0]
        
        # Start iterating
        for obs_id in tqdm(range(num_test_obs), leave=False, desc='DRNet: Evaluate test observations'):
            # Get observation
            observation = tData.x[obs_id]
            # Repeat observation to get x
            x = observation.repeat(num_integration_samples, 1)
            
            true_outcomes = [get_outcome(observation, tData.v, d, self.trainDataset['response']) for d in treatment_strengths]
            true_outcomes = np.array(true_outcomes)
            
            # Predict outcomes
            pred_outcomes = np.array(self.predictObservation(x, treatment_strengths)).reshape(1,-1).squeeze()
            
            # MISE #
            # Calculate MISE for dosage curve
            mise = romb((pred_outcomes - true_outcomes) ** 2, dx=step_size)
            mises.append(mise)
            
            # PE #
            # Find best treatment strength
            best_pred_d = treatment_strengths[np.argmax(pred_outcomes)].detach()
            best_actual_d = treatment_strengths[np.argmax(true_outcomes)].detach()
            
            # Get policy error by comparing best predicted and best actual dosage
            policy_error = (best_pred_d - best_actual_d) ** 2
            pes.append(policy_error)
            
            # FE #
            fact_outcome = dataset['y'][obs_id]
            # Find closest dosage in treatment strengths
            fact_d = dataset['d'][obs_id]
            lower_id = np.searchsorted(treatment_strengths, fact_d, side="left") - 1
            upper_id = lower_id + 1
            
            lower_d = treatment_strengths[lower_id]
            upper_d = treatment_strengths[upper_id]
            
            lower_est = pred_outcomes[lower_id]
            upper_est = pred_outcomes[upper_id]
            
            # Get calc as linear interpolation
            pred_outcome = ((fact_d - lower_d) * upper_est + (upper_d - fact_d) * lower_est) / (upper_d - lower_d)
            
            fe = ((fact_outcome - pred_outcome) ** 2)
            fes.append(fe)
            
        return np.sqrt(np.mean(mises)), np.sqrt(np.mean(pes)), np.sqrt(np.mean(fes))