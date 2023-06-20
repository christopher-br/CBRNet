# Written by Christopher Bockel-Rickermann. Copyright (c) 2022

# Load necessary modules
import numpy as np
from tqdm import tqdm

from scipy.integrate import romb

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

import torch

from modules.DataGen import get_outcome

from sklearn.decomposition import PCA

from modules.DataGen import torchDataset

class LinReg():
    """
    The LinReg model.
    
    Attributes:
        config (dict): A dictionary with all parameters to create and train the model
        dataset (dict): Dataset for training
        
    Config entries:
        'polyDegree' (int): Degree of polynomials used to model the output
        'penalty' (str): The penalty used for training
    """
    def __init__(self, config, dataset):
        """
        The LinReg model.
        
        Parameters:
            config (dict): A dictionary with all parameters to create and train the model
            dataset (dict): Dataset for training
            
        Config entries:
            'polyDegree' (int): Degree of polynomials used to model the output
            'penalty' (str): The penalty used for training
        """
        np.random.seed(42)
        
        # Save training data
        self.trainDataset = dataset
        
        # Save settings
        self.penalty = config.get('penalty')
        self.poly_degree = config.get('polyDegree')
        
        # Define preprocessor
        self.feat_transform = PolynomialFeatures(self.poly_degree)
        
        # Train model
        self.trainModel(dataset)
        
    def trainModel(self, dataset):
        # Define PCA
        self.pca = PCA(50, random_state=42).fit(dataset['x'])
        
        # Reduce dim of x
        train_x = self.pca.transform(dataset['x'])
        
        # Define arrays
        train_x = np.column_stack((train_x,
                                   dataset['d'].reshape((-1,1))))
        train_y = dataset['y']
        
        # Transform data
        train_x = self.feat_transform.fit_transform(train_x)
        
        # Ini model
        if self.penalty == False:
            self.model = sm.OLS(train_y, train_x).fit()
        else:
            self.model = sm.OLS(train_y, train_x).fit_regularized()
        
    def validateModel(self, dataset):
        # Reduce dim of x
        val_x = self.pca.transform(dataset['x'])
        # Define arrays
        val_x = np.column_stack((val_x,
                                 dataset['d'].reshape((-1,1))))
        val_y = dataset['y']
        
        # Transform data
        val_x = self.feat_transform.fit_transform(val_x)
        
        preds = self.model.predict(val_x)
        
        val_mse = np.mean((preds - val_y) ** 2)
        
        return val_mse
        
    def predictObservation(self, x, d):
        # Transform inputs
        obs = np.column_stack((x,d))
        
        obs_x = self.feat_transform.fit_transform(obs)
        
        outcomes = self.model.predict(obs_x)
        
        return outcomes
        
    def getDR(self, observation):
        # Save observation as torch tensor
        observation = torch.Tensor(observation)
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Repeat observation
        x = observation.repeat(num_integration_samples, 1).numpy()
        
        # PCA transform
        x = self.pca.transform(x)
        
        # Predict dr curve
        dr_curve = self.predictObservation(x, treatment_strengths)
        
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
        Generates MISE and PE values for a LinReg model on a torchDataset
        
        Parameters:
            dataset (dict): A dataset
        """
        tData = torchDataset(dataset)
        # Define arrays
        test_x = np.column_stack((dataset['x'],
                                  dataset['d'].reshape((-1,1))))
        test_y = dataset['y']
        
        # Initialize result arrays
        mises = []
        pes = []
        fes = []
        
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 1 / num_integration_samples
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Get number of observations
        num_test_obs = dataset['x'].shape[0]
        
        # PCA transform x
        tData_x_pca = self.pca.transform(tData.x)
        
        # Start iterating
        for obs_id in tqdm(range(num_test_obs), leave=False, desc='LinReg: Evaluate test observations'):
            # Get observation
            observation = tData.x[obs_id]
            observation_pca = torch.Tensor(tData_x_pca[obs_id])
            # Repeat observation to get x
            x = observation_pca.repeat(num_integration_samples, 1).numpy()
            
            true_outcomes = [get_outcome(observation, tData.v, d, self.trainDataset['response']) for d in treatment_strengths]
            true_outcomes = np.array(true_outcomes)
            
            # Predict outcomes
            pred_outcomes = np.array(self.predictObservation(x, treatment_strengths))
            
            ## MISE ##
            # Calculate MISE for dosage curve
            mise = romb((pred_outcomes - true_outcomes) ** 2, dx=step_size)
            mises.append(mise)
            
            # PE #
            # Find best treatment strength
            best_pred_d = treatment_strengths[np.argmax(pred_outcomes)]
            best_actual_d = treatment_strengths[np.argmax(true_outcomes)]
            
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