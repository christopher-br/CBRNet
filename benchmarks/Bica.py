# Code ispired by Bica et al. (2020)

import numpy as np
import torch

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from torch.utils.data import Dataset

from utils.Responses import get_outcome
from utils.Biases import beta_mode


def gen_bica_dataset(args, x, n_clusters=3):
    """
    Function to generate train, validation, and test datasets
    
    Parameters:
        args (dict): Arguments to generate the dataset:
        x (np array): An array containing all x variables needed for training. If no array passed, data will follow n-dimensional normal distribution
        
    args entries:
        'bias' (float): The amount of bias [0, inf)
        'responseType' (str): Which response to use for generating the observation
        'binaryTargets' (bool): If observed outcomes are a binary realization of otherwise continuous outcomes
        'testFraction' (float): The amount of observations used for testing
        'valFraction' (float): The amount of observations used for validation
        'noise' (float): The standard deviation of the error term
        'numObs' (int): The number of observations to be sampled
        'numVars' (int): The number of variables in the data
        'seed' (int): A seed for random number generation
    """
    # Set seed
    np.random.seed(args['seed'])
    
    ## selection_bias = args['dosageSelectionBias']
    ## distribution_bias = args['dosageDistributionBias']
    bias = args['bias']
    
    # Generate placeholder
    dataset = dict()
    
    # Save x
    # Get cluster
    cluster = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto').fit(x).labels_
    
    # Normalize x
    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    for i in range(x.shape[0]):
        x[i] = x[i] / np.linalg.norm(x[i])
        
    # Assign x
    dataset['x'] = x
    
    # Assign clusters
    dataset['c'] = cluster
    
    # Get random cluster modes
    ## n_cluster = len(np.unique(cluster))
    ## if selection_bias == 'random':
    ##     dosage_modes_per_cluster = np.random.rand(n_cluster)
    ## else:
    ##     lb = 0.5 - 0.5 * selection_bias
    ##     ub = 0.5 + 0.5 * selection_bias
    ##     dosage_modes_per_cluster = np.linspace(lb,ub,n_cluster)
    
    # Generate rest
    dataset['y'] = []
    dataset['d'] = []
    dataset['v'] = np.zeros(shape=(5,args['numVars']))
    
    # Normalize x
    dataset['x'] = (dataset['x'] - np.min(dataset['x'], axis=0)) / (np.max(dataset['x'], axis=0) - np.min(dataset['x'], axis=0))
    for i in range(dataset['x'].shape[0]):
        dataset['x'][i] = dataset['x'][i] / np.linalg.norm(dataset['x'][i])
    
    # Generate weights
    for i in range(5):
        dataset['v'][i] = np.random.uniform(0,1,size=(args['numVars']))
        dataset['v'][i] = dataset['v'][i] / np.linalg.norm(dataset['v'][i])
        
    # Generate observations
    for obs_id in tqdm(range(dataset['x'].shape[0]), leave=False, desc='Generate synthetic targets'):
        d, y = generate_observation(x=dataset['x'][obs_id],
                                    v=dataset['v'],
                                    ## distribution_bias=distribution_bias,
                                    bias=bias,
                                    noise_std=args['noise'],
                                    response=args['responseType']## ,
                                    )
                                    ## cluster_mode=dosage_modes_per_cluster[cluster[obs_id]])
        # Append to list
        dataset['d'].append(d)
        # If 'binaryTargets' == True
        if args['binaryTargets'] == True:
            dataset['y'].append(float(np.random.binomial(1,y,1)))
        else:
            dataset['y'].append(y)
            
    # Transform to np array
    for key in ['y', 'd', 'c']:
        dataset[key] = np.array(dataset[key])
        
    # Set d to (0,1)
    # dataset['d'] = (dataset['d'] - np.min(dataset['d'])) / (np.max(dataset['d']) - np.min(dataset['d']))
        
    # Get split indices
    train_idx, val_idx, test_idx = get_splits(dataset['x'].shape[0],
                                              args['testFraction'],
                                              args['valFraction'])
    
    # Generate sub-datasets
    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    
    # Save data
    for key in ['x', 'y', 'd', 'c']:
        dataset_train[key] = dataset[key][train_idx]
        dataset_val[key] = dataset[key][val_idx]
        dataset_test[key] = dataset[key][test_idx]
    
    # Save weights 
    dataset_train['v'] = dataset['v']
    dataset_val['v'] = dataset['v']
    dataset_test['v'] = dataset['v']
    
    # Save type
    dataset['response'] = args['responseType']
    dataset_train['response'] = args['responseType']
    dataset_val['response'] = args['responseType']
    dataset_test['response'] = args['responseType']
    
    return dataset, dataset_train, dataset_val, dataset_test
        

class torchDataset(Dataset):
    """
    The torchDataset object that can be fed to a pytorch dataloader.
    
    Attributes:
        dataset (dict): A dataset in dictionary form:
        
    dataset entries:
        'x' (np array): Observations
        'y' (np array): Observed outcomes for every observation in x
        'd' (np array): A dosage for every observation in x
        'v' (np array): A weight vector of the size of x
    """
    def __init__(self, dataset, gen_pca=False):
        """
        The constructor of the torchDataset object
        
        Parameters:
            dataset (dict): A dataset in dictionary form:
            
        dataset entries:
            'x' (np array): Observations
            'y' (np array): Observed outcomes for every observation in x
            'd' (np array): A dosage for every observation in x
            'v' (np array): A weight vector of the size of x
        """
        # Save v vector
        self.v = dataset['v']
        # Save clusters
        ## self.c = dataset['c']
        # Assign values according to indices    
        self.x = torch.from_numpy(dataset['x']).type(torch.float32)
        self.y = torch.from_numpy(dataset['y']).type(torch.float32)
        self.d = torch.from_numpy(dataset['d']).type(torch.float32)
        # Calculate PCA values
        if gen_pca:
            k = np.min((dataset['x'].shape[1], 500))
            x_pca = PCA(k, random_state=42).fit(dataset['x']).transform(dataset['x'])
            self.x_pca = torch.from_numpy(x_pca).type(torch.float32)
        else:
            self.x_pca = self.x
        # Save length
        self.length = dataset['x'].shape[0]
        # Save response type
        self.response = dataset['response']
    
    # Define necessary fcts
    def get_data(self):
        """
        Gets the entire data in the torchDataset object
        """
        return self.x, self.y, self.d
    
    def get_pca_data(self):
        """
        Gets the pca-transformed x values from the torchDataset objectr
        """
        return self.x_pca
    
    def __getitem__(self, index):
        """
        Gets a subset of the data in the torchDataset object
        """
        return self.x[index], self.y[index], self.d[index]
    
    def __len__(self):
        """
        Gets the length of the data in the torchDataset object
        """
        return self.length
    
    
def generate_observation(x, v, bias, noise_std, response):
    """
    Generates an observation
    
    Parameters:
        x (np array): The observation
        v (np.array): The linear coefficients to calculate dose response parameters
        distribution_bias (float): The strength of the distribution bias
        noise_std (float): The size of the normally distributed error term
        response (str): Which response to use
        bias_type (str): Which bias type to use
        cluster_mode (float): The modal dosage for all observations of the cluster
    """
    # Save alpha (1+bias)
    a = 1 + bias
    
    # Clac mode
    mode = np.dot(x, v[4])
    
    # Calculate beta
    b = beta_mode(a, mode=mode)
    
    # Sample the dosage
    d = np.random.beta(a, b)
    
    # Get the outcome
    y = get_outcome(x, v, d, response)

    # Assign error
    y = (y + np.random.normal(0, noise_std))

    return d, y
    
def get_splits(num_obs, val_frac, test_frac):
    """
    Gets indices for generating test, train, and test data of defined size
    
    Parameters:
        num_obs (int): Number of observations in dataset
        val_frac (float): Share of data to be used for validation
        test_frac (float): Share of data to be used for testing
    """
    # Get indexes
    indexes = [i for i in range(num_obs)]
    # Get number of observations per test/val
    num_test = int(num_obs * test_frac)
    num_val = int(num_obs * val_frac)
    
    # Get indexes
    rest_idx, test_idx = train_test_split(indexes, test_size=num_test)
    train_idx, val_idx = train_test_split(rest_idx, test_size=num_val)
    
    return train_idx, val_idx, test_idx