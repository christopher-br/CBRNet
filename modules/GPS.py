# Load necessary modules
import numpy as np
from tqdm import tqdm

from scipy.integrate import romb

import torch

from modules.DataGen import get_outcome

from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.stats import norm

from sklearn.decomposition import PCA

from modules.DataGen import torchDataset

class GPS():
    """
    The GPS model.
    
    Attributes:
        config (dict): A dictionary with all parameters to create and train the model
        dataset (dict): Dataset for training
        
    Config entries:
        'treatPolyDegree' (int): Highest degree of polynomials considered for the treatment model
        'outcomePolyDegree' (int): Highest degree of polynomials considered for the outcome model
    """
    def __init__(self, config, dataset):
        """
        The GPS model.
        
        Parameters:
            config (dict): A dictionary with all parameters to create and train the model
            dataset (dict): Dataset for training
            
        Config entries:
            'treatPolyDegree' (int): Highest degree of polynomials considered for the treatment model
            'outcomePolyDegree' (int): Highest degree of polynomials considered for the outcome model
        """
        np.random.seed(42)
        
        # Save config
        self.config = config        
        
        # Save training data
        self.trainDataset = dataset
        
        # Save settings
        self.treat_poly_degree = config.get('treatPolyDegree')
        self.outcome_poly_degree = config.get('outcomePolyDegree')
        
        # Define feature transformer
        self.treat_feat_transform = PolynomialFeatures(self.treat_poly_degree)
        self.outcome_feat_transform = PolynomialFeatures(self.outcome_poly_degree)
        
        self.trainModel(dataset)
        
    def trainModel(self, dataset):
        # Define arrays
        train_x = dataset['x']
        train_d = dataset['d']
        train_y = dataset['y']
        
        # Define PCA
        self.pca = PCA(50, random_state=42).fit(train_x)
        
        # Reduce dim of train_x
        train_x = self.pca.transform(train_x)
        
        # Transform according to settings
        train_x = self.treat_feat_transform.fit_transform(train_x)
        
        # Calc model of dosage as function of x
        self.model_d = sm.OLS(train_d, train_x).fit()
        
        # Estimate distribution of errors
        errors = self.model_d.predict(train_x) - train_d
        
        # Calc std of errors
        _, self.std_d = norm.fit(errors)
        
        # Calculate prop score
        prop_score = norm.pdf(errors, loc=0, scale=self.std_d)
        
        # Create training data for outcome model
        train_gps = np.column_stack((dataset['d'],
                                     prop_score))
        
        train_gps = self.outcome_feat_transform.fit_transform(train_gps)
        
        # Calc final model
        self.model_y = sm.OLS(train_y, train_gps).fit()
        
    def validateModel(self, dataset):
        # Define arrays
        val_x = dataset['x']
        val_d = dataset['d']
        val_y = dataset['y']
        
        # Reduce dim
        val_x = self.pca.transform(val_x)
        
        # Transform according to settings
        val_x = self.treat_feat_transform.fit_transform(val_x)
        
        # Get errors
        errors = self.model_d.predict(val_x) - val_d
        
        # Get prop score
        prop_score = norm.pdf(errors, loc=0, scale=self.std_d)
        
        # Get data for outcome model
        val_gps = np.column_stack((dataset['d'],
                                   prop_score))
        
        val_gps = self.outcome_feat_transform.fit_transform(val_gps)
        
        # Predict outcomes
        preds = self.model_y.predict(val_gps)
        
        # Get val MSE
        val_mse = np.mean((preds - val_y) ** 2)
        
        return val_mse
        
    def predictObservation(self, x, d):
        # Transform inputs
        obs = self.treat_feat_transform.fit_transform(x)
        
        # Get errors
        errors = self.model_d.predict(obs) - d
        
        # Get prop scores
        prop_score = norm.pdf(errors, loc=0, scale=self.std_d)
        
        # Transform d
        d = d.reshape(-1,1)
        
        # Get data for final model
        obs_gps = np.column_stack((d,
                                   prop_score))
        
        obs_gps = self.outcome_feat_transform.fit_transform(obs_gps)
        
        # Predict outcomes
        outcomes = self.model_y.predict(obs_gps)
        
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
        Generates MISE and PE values for a GPS model on a torchDataset
        
        Parameters:
            dataset (dict): A dataset
        """
        tData = torchDataset(dataset)
        # Define arrays
        test_x = dataset['x']
        test_d = dataset['d']
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
        for obs_id in tqdm(range(num_test_obs), leave=False, desc='GPS: Evaluate test observations'):
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
    