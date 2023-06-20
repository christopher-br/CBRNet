# Load necessary modules
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scienceplots

from tqdm import tqdm
from scipy.integrate import romb
from scipy.stats import gaussian_kde, norm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from utils.Responses import get_outcome
from utils.Loss import mmd_lin, mmd_rbf, mmd_none, mmd_lin_w, mmd_rbf_w
from utils.Binning import lin_bins, quantile_bins, jenks_bins, kde_bins, kmeans_bins

from modules.DataGen import torchDataset

class CBRNet(pl.LightningModule):
    """
    The CBRNets model with a validation error weighted by a GPS (generalized propensity score) estimate.
    
    Attributes:
        config (dict): A dictionary with all parameters to create and train the model
        dataset (dict): Dataset for training
        
    Config entries:
        'learningRate' (float): The learning rate
        'batchSize' (int): The batch size
        'numSteps' (int): The number of training steps
        'numRepresentationLayers' (int): The number of layers for representation learning
        'numInferenceLayers' (int): The number of layers for inference
        'numBins' (int): The number of individual bins
        'inputSize' (int): The input size
        'hiddenSize' (int): The size of each hidden layer 
        'regularizeBeta' (float): Strength of divergence regularization
        'activationFct' (str): The activation function to use in the forward pass
        'mmdType' (str): Type of MMD regularization
        'verbose' (bool): Verbosity during network training
        'binningMethod' (str): Which method to use to do binning
        'weightedAvg' (bool): Whether mmd is a weighted average of different groups or not
        'accelerator' (str): Which accelerator to use, can be 'cpu' or 'mps'
    """
    
    def __init__(self, config, dataset):
        """
        The constructor of the CBRNets_Model class.
        
        Parameters:
            config (dict): A dictionary with all parameters to create and train the model
            dataset (dict): Dataset for training
            
        Config entries:
            'learningRate' (float): The learning rate
            'batchSize' (int): The batch size
            'numSteps' (int): The number of training steps
            'numRepresentationLayers' (int): The number of layers for representation learning
            'numInferenceLayers' (int): The number of layers for inference
            'numBins' (int): The number of individual bins
            'inputSize' (int): The input size
            'hiddenSize' (int): The size of each hidden layer 
            'regularizeBeta' (float): Strength of divergence regularization
            'activationFct' (str): The activation function to use in the forward pass
            'mmdType' (str): Type of MMD regularization
            'verbose' (bool): Verbosity during network training
            'binningMethod' (str): Which method to use to do binning
            'weightedAvg' (bool): Whether mmd is a weighted average of different groups or not
            'accelerator' (str): Which accelerator to use, can be 'cpu' or 'mps'
        """
        # Ini the super module
        super(CBRNet, self).__init__()
        
        torch.manual_seed(42)
        
        # Save config
        self.config = config
        
        # Save training data
        self.t_trainDataset = torchDataset(dataset)
        self.trainDataset = dataset
        
        # Save settings
        self.learningRate = config.get('learningRate')
        self.batch_size = config.get('batchSize')
        self.num_steps = config.get('numSteps')
        self.num_r_layers = config.get('numRepresentationLayers')
        self.num_i_layers = config.get('numInferenceLayers')
        self.num_bins = config.get('numBins')
        self.input_size = config.get('inputSize')
        self.hidden_size = config.get('hiddenSize')
        self.beta = config.get('regularizeBeta')
        self.activation = config.get('activationFct')
        self.mmd = config.get('mmdType')
        self.verbose = config.get('verbose')
        self.accelerator = config.get('accelerator')
        self.binning_method = config.get('binningMethod')
        
        # Save activation functions in dict
        self.activation_map = {'linear': nn.Identity(),
                               'elu': nn.ELU(),
                               'relu': nn.ReLU(),
                               'leaky relu': nn.LeakyReLU(),
                               'sigmoid': nn.Sigmoid()}
        
        # Save map of different regularization fcts
        self.mmd_map = {'linear': mmd_lin,
                        'rbf': mmd_rbf,
                        'weighted linear': mmd_lin_w,
                        'weighted rbf': mmd_rbf_w,
                        'none': mmd_none}
        
        # Save map of different binning fcts
        self.binning_map = {'linear': lin_bins,
                            'quantile': quantile_bins,
                            'jenks': jenks_bins,
                            'kde': kde_bins,
                            'kmeans': kmeans_bins}
        
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
        obs_binned, binner, n_bins = self.binning_map[self.binning_method](d=self.t_trainDataset.d,
                                                                           x=self.t_trainDataset.x,
                                                                           n_bins=self.num_bins)
        
        # Save binner fct to self
        self.binner = binner
        # Overwrite num_bins (only makes change for binning fct that calculated optiomal number of bins)
        self.num_bins = n_bins
        
        # Calculate weights per bin
        # Save obs_binned as tensor
        obs_binned_tensor = torch.Tensor(obs_binned).int()
        # Get weight as inverse to population size in bin
        self.obs_weight_inv = 1 / torch.bincount(obs_binned_tensor, minlength=self.num_bins)
        self.obs_weight_inv = self.obs_weight_inv / torch.sum(self.obs_weight_inv)
        # Get obs weight
        self.obs_weight = torch.bincount(obs_binned_tensor, minlength=self.num_bins)
        self.obs_weight = self.obs_weight / torch.sum(self.obs_weight)
        
        # Structure
        # Representation learning layers
        # Initialize layers
        self.representation_layers = nn.Sequential(nn.Linear(self.input_size, self.hidden_size))
        self.representation_layers.append(self.activation_map[self.activation])
        # Add additional layers
        for i in range(self.num_r_layers - 1):
            self.representation_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.representation_layers.append(self.activation_map[self.activation])
        
        # Head layers
        self.head_layers = nn.Sequential(nn.Linear(self.hidden_size + 1, self.hidden_size))
        self.head_layers.append(self.activation_map[self.activation])
        # Add additional hidden layers
        for i in range(self.num_i_layers - 1):
            self.head_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.head_layers.append(self.activation_map[self.activation])
        # Add output layer
        self.head_layers.append(nn.Linear(self.hidden_size, 1))
            
        # Train
        self.trainModel(self.trainDataset)
        
                       
    # Define forward step
    def forward(self, x, d):
        """
        The forward function of a CBRNets model.
        
        Parameters:
            x (torch tensor): A torch tensor (usually passed over by a Dataloader)
            d (torch tensor): A torch tensor (usually passed over by a Dataloader)
        """
        # Feed through shared layers
        x = self.representation_layers(x)
        
        # Reshape d
        d = d.reshape(-1,1)
        
        # Add d to x
        res = torch.cat((x, d), dim=1)
        
        # Results array
        res = self.head_layers(res)
            
        # Return
        return res, x
    
    # Training step
    def training_step(self, batch, batch_idx):
        """
        The training step function of a CBRNets model.
        
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
        y, hidden = self(x, d)
        
        # Squeeze inputs
        y = y.squeeze()
        
        # Get the loss
        loss_mse = F.mse_loss(y, y_true)
        
        loss_div = self.mmd_map[self.mmd](hidden=hidden, 
                                          x=x, 
                                          d=d, 
                                          binner=self.binner, 
                                          n_cluster=self.num_bins, 
                                          weight_v=self.obs_weight_inv)
        
        # Total loss
        loss = loss_mse + self.beta * loss_div
        
        return loss
    
    def configure_optimizers(self):
        """
        The optimizer of a CBRNets model
        """
        return torch.optim.Adam(self.parameters(), lr=self.learningRate)
    
    def dataloader(self, torchDataset, shuffle=True):
        """
        Generates a DataLoader for a CBRNets model
        
        Parameters:
            torchDataset (torchDataset): A torchDataset
            shuffle (bool): An indicator if data should be shuffled everytime Dataloader is called
        """
        # Generate DataLoader
        return DataLoader(torchDataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def trainModel(self, dataset):
        """
        Fits a CBRNets model to a dataset
        
        Parameters:
            dataset (dict): Dataset for training
        """
        # GPS
        # Define PCA
        self.pca = PCA(50, random_state=42).fit(dataset['x'])
        # Reduce dim of x
        x_pca = self.pca.transform(dataset['x'])
        # Add intercept
        x_pca = PolynomialFeatures(1).fit_transform(x_pca)
        
        # Train components for GPS model
        self.dosage_model = sm.OLS(dataset['d'], x_pca).fit()
        errors = self.dosage_model.predict(x_pca) - dataset['d']
        _, self.dosage_std = norm.fit(errors)
        
        # Model
        tData = torchDataset(dataset)
        # Define loader
        loader = self.dataloader(tData)
        # Fit
        self.trainer.fit(self, loader) 
        
    def validateModel(self, dataset):
        """
        Validates a CBRNets model on a dataset, by calculating the mean error on the observations in the dataset
        
        Parameters:
            dataset (dict): Dataset for training
        """
        # Calc GPS
        x_pca = self.pca.transform(dataset['x'])
        x_pca = PolynomialFeatures(1).fit_transform(x_pca)
        errors = self.dosage_model.predict(x_pca) - dataset['d']
        # Clip 5% outliers
        clip_val = norm.ppf(0.975, loc=0, scale=self.dosage_std)
        errors = np.clip(errors, a_min=-clip_val, a_max=clip_val)
        prop_scores = torch.Tensor(
            norm.pdf(errors, loc=0, scale=self.dosage_std)
        )
        # Get weights as inverse of Prop score
        weights = 1 / prop_scores
        
        # Get val score
        tData = torchDataset(dataset)
        # Get true outcomes
        x, y_true, d = tData.get_data()
        
        # Generate predictions
        y = torch.Tensor(self.predictObservation(x, d).squeeze())
        
        # Get mse
        val_mse = (torch.sum(weights * ((y - y_true) ** 2)) / torch.sum(weights)).detach().item()
        
        # Return mse
        return val_mse
    
    def predictObservation(self, x, d):
        """
        Generates predictions of a CBRNets model for data
        
        Parameters:
            x (torch tensor): A torch tensor with observations
            d (torch tensor): A torch tensor with dosage information
        """
        # Get outcomes (array form)
        outcomes, _ = self(x, d)
        
        # Squeeze
        predictions = outcomes.squeeze().detach().numpy()
            
        return predictions
    
    def getDR(self, observation):
        """
        Generates the predicted dose response of a CBRNets model for a single observation
        
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
        Generates the true dose response of a CBRNets model for a single observation
        
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
        for obs_id in tqdm(range(num_test_obs), leave=False, desc='CBRNet: Evaluate test observations'):
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
    
    
    def plotTsne(self, dataset, seed=42, perplexity=5, style='science', scaling_factor=50, save=False, alpha=1.):
        """
        Generates tSNE plot of ingoing x and hidden representation of x
        
        Parameters:
            dataset (dict): Dataset for training
        """
        tData = torchDataset(dataset)
        # Ini TSNE
        tsne = TSNE(2,random_state=seed,perplexity=perplexity)
        
        # Get dosages
        d = tData.d.detach().numpy()
        
        # Get clusters
        c = tData.c
        
        # Get input space
        input_x = tData.x
        
        # Get latent
        _, hidden_x = self(tData.x, tData.d)
        hidden_x = hidden_x.detach().numpy()
        
        # Generate TSNE
        tsne_input_x = tsne.fit_transform(input_x)
        tsne_hidden_x = tsne.fit_transform(hidden_x)
        
        # Generate colors
        colors = np.unique(c) - np.min(np.unique(c))
        colors = colors / np.max(colors)
        
        # Plot
        plt.style.use(style)
        
        # dose distribution
        fig_doses = plt.figure(figsize=(3,2.5))
        dose_data = [d[c==i] for i in np.unique(c)]
        # Generate scaling factors for kde
        bins = np.linspace(0,1,50)
        kde_scaling = [np.histogram(dose_data[i], bins=bins)[0].max() for i in np.unique(c)]
        
        plt.hist(dose_data,
                 bins,
                 color=cm.prism(colors),
                 range=(0,1),
                 rwidth=1.)
        plt.xlabel('Dose')
        plt.ylabel('Frequency')
        
        dose_samples = np.linspace(0,1,100)
        for i,j in enumerate(colors):
            plt.plot(dose_samples,
                     gaussian_kde(dose_data[i])(dose_samples)*(1/np.max(gaussian_kde(dose_data[i])(dose_samples)))*kde_scaling[i],
                     color=cm.prism(j),
                     linewidth=1.5)

        
        # input
        fig_input = plt.figure(figsize=(3,2.5))
        plt.scatter(x=tsne_input_x[:,0],
                    y=tsne_input_x[:,1],
                    c=c,
                    s=d*scaling_factor,
                    edgecolors='black',
                    linewidths=0.1,
                    cmap=cm.prism,
                    alpha=alpha)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # hidden
        fig_hidden = plt.figure(figsize=(3,2.5))
        plt.scatter(x=tsne_hidden_x[:,0],
                    y=tsne_hidden_x[:,1],
                    c=c,
                    s=d*scaling_factor,
                    edgecolors='black',
                    linewidths=0.1,
                    cmap=cm.prism,
                    alpha=alpha)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        if save:
            fig_doses.savefig('dose_distr.pdf')
            fig_input.savefig('input_x.pdf')
            fig_hidden.savefig('hidden_x.pdf')
        
        return fig_doses, fig_input, fig_hidden
    