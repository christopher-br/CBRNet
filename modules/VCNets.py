# Written by Christopher Bockel-Rickermann. Copyright (c) 2022

# Code adapted from:
# - Nie et al. (2021):
#       Vcnet and functional targeted regularization 
#       for learning causal effects of continuous treatments

# Load necessary modules
import numpy as np
from tqdm import tqdm

from scipy.integrate import romb

from modules.DataGen import get_outcome

import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import math

from modules.DataGen import torchDataset

from sklearn.decomposition import PCA

class VCNet():
    """
    The VCNet model.
    
    Attributes:
        config (dict): A dictionary with all parameters to create and train the model
        dataset (dict): Dataset for training
        
    Config entries:
        'learningRate' (float): The learning rate
        'batchSize' (int): The batch size
        'numEpochs' (int): The number of training epochs
        'inputSize' (int): The input size
        'hiddenSize' (int): The size of each hidden layer
        'numGrid' (int): Grid size
        'knots' (array): Knots for regularization
        'degree' (int): Degree of the power function
        'targetReg' (bool): Indicator if traget reg is be used
    """
    def __init__(self, config, dataset):
        """
        The constructor of the VCNet model.
        
        Parameters:
            config (dict): A dictionary with all parameters to create and train the model
            dataset (dict): Dataset for training
            
        Config entries:
            'learningRate' (float): The learning rate
            'batchSize' (int): The batch size
            'numEpochs' (int): The number of training epochs
            'inputSize' (int): The input size
            'hiddenSize' (int): The size of each hidden layer
            'numGrid' (int): Grid size
            'knots' (array): Knots for regularization
            'degree' (int): Degree of the power function
            'targetReg' (bool): Indicator if traget reg is be used
        """
        
        torch.manual_seed(42)
        
        # Save config
        self.config = config
        
        # Save training data
        self.trainDataset = dataset
        
        # Save settings
        self.num_pc = 50
        self.learningRate = config.get('learningRate')
        self.batch_size = config.get('batchSize')
        self.num_epoch = config.get('numEpochs')
        self.input_size = self.num_pc # PCA data used, otherwise config.get('inputSize')
        self.hidden_size = config.get('hiddenSize')
        self.num_grid = config.get('numGrid')
        self.knots = config.get('knots')
        self.degree = config.get('degree')
        self.target_reg = config.get('targetReg')
        
        # Train model
        self.trainModel(dataset)

    def trainModel(self, dataset):
        wd = 5e-3
        momentum = 0.9
        # targeted regularization optimizer
        tr_wd = 5e-3
        num_epoch = self.num_epoch
        
        # Define PCA
        self.pca = PCA(self.num_pc, random_state=42).fit(dataset['x'])
        
        # Reduce dim of train_x
        train_x = self.pca.transform(dataset['x'])
        
        # Get training data
        train_matrix = torch.from_numpy(
            np.column_stack((
                dataset['d'],
                train_x,
                dataset['y']
            ))).float()
        
        # Define loader
        train_loader = get_iter(train_matrix, self.batch_size, shuffle=True)
        
        # Define settings
        cfg_density = [
            (self.input_size, self.hidden_size, 1, 'relu'),
            (self.hidden_size, self.hidden_size, 1, 'relu')
        ]
        num_grid = self.num_grid
        cfg = [
            (self.hidden_size, self.hidden_size, 1, 'relu'),
            (self.hidden_size, 1, 1, 'id')
        ]
        degree = self.degree
        knots = self.knots
        
        # Load model
        self.model = VCNet_module(cfg_density, num_grid, cfg, degree, knots)
        self.model._initialize_weights()
        
        if self.target_reg:
            tr_knots = list(np.arange(0.1, 1, 0.1))
            tr_degree = 2
            self.TargetReg = TR(tr_degree, tr_knots)
            self.TargetReg._initialize_weights()
            
        # Get best settings
        init_lr = 0.0001
        alpha = 0.5
        tr_init_lr = 0.001
        beta = 1.
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=init_lr,
            momentum=momentum,
            weight_decay=wd,
            nesterov=True
        )

        if self.target_reg:
            tr_optimizer = torch.optim.SGD(
                self.TargetReg.parameters(), 
                lr=tr_init_lr, 
                weight_decay=tr_wd
            )
            
        # Train
        for epoch in tqdm(range(num_epoch), leave=False, desc='Train VCNet'):
            for idx, (inputs, y) in enumerate(train_loader):
                idx = idx
                t = inputs[:, 0]
                x = inputs[:, 1:]
                
                # Train with target reg
                if self.target_reg:
                    optimizer.zero_grad()
                    out = self.model.forward(t, x)
                    trg = self.TargetReg(t)
                    loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                    loss.backward()
                    optimizer.step()

                    tr_optimizer.zero_grad()
                    out = self.model.forward(t, x)
                    trg = self.TargetReg(t)
                    tr_loss = criterion_TR(out, trg, y, beta=beta)
                    tr_loss.backward()
                    tr_optimizer.step()
                # Train withouth target reg
                else:
                    optimizer.zero_grad()
                    out = self.model.forward(t, x)
                    loss = criterion(out, y, alpha=alpha)
                    loss.backward()
                    optimizer.step()
        
    def validateModel(self, dataset):
        # Reduce dim of train_x
        val_x = self.pca.transform(dataset['x'])
        # Get validation data
        val_matrix = torch.from_numpy(
            np.column_stack((
                dataset['d'],
                val_x,
                dataset['y']
            ))).float()
        
        # Define val loader
        val_loader = get_iter(val_matrix, val_matrix.shape[0], shuffle=False)
        
        for idx, (inputs, y) in enumerate(val_loader):
            # Get inputs
            t = inputs[:, 0]
            x = inputs[:, 1:]
            
            # Get estimates
            out = self.model.forward(t, x)[1].data.squeeze()
            y = y.data.squeeze()
            
            # Get MSE
            val_mse = F.mse_loss(out, y).detach().item()
            
            return val_mse
        
    def predictObservation(self, x, d):
        # Get pred data
        pred_matrix = torch.from_numpy(np.column_stack((d,x,d))).float()
        
        # Define pred loader
        pred_loader = get_iter(pred_matrix, pred_matrix.shape[0], shuffle=False)
        
        for idx, (inputs, y) in enumerate(pred_loader):
            # Get inputs
            t = inputs[:, 0]
            x = inputs[:, 1:]
            
            # Get estimates
            out = self.model.forward(t, x)[1].data.squeeze()
            
        return out
        
    def getDR(self, observation):
        # Save observation as torch tensor
        observation = torch.Tensor(observation)
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = torch.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Repeat observation
        x = observation.repeat(num_integration_samples, 1)
        
        # PCA transform
        x = self.pca.transform(x)
        
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
        treatment_strengths = torch.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Get number of observations
        num_test_obs = dataset['x'].shape[0]
        
        # PCA transform x
        tData_x_pca = self.pca.transform(test_x)
        
        # Start iterating
        for obs_id in tqdm(range(num_test_obs), leave=False, desc='VCNet: Evaluate test observations'):
            # Get observation
            observation = test_x[obs_id]
            observation = torch.Tensor(observation)
            observation_pca = torch.Tensor(tData_x_pca[obs_id])
            
            # Repeat
            x = observation_pca.repeat(num_integration_samples, 1)
            
            true_outcomes = [get_outcome(observation, dataset['v'], d, dataset['response']) for d in treatment_strengths]
            true_outcomes = np.array(true_outcomes)
            
            # Predict outcomes
            pred_outcomes = np.array(self.predictObservation(x, treatment_strengths)).reshape(1,-1).squeeze()
            
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
    
#########
# Utils #
#########

########
# Data #
########

class Dataset_from_matrix(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])

def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

########
# Eval #
########

def curve(model, test_matrix, t_grid, targetreg=None):
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg is None:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            out = out[1].data.squeeze()
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse
    else:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse

#########
# Model #
#########

class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out

class VCNet_module(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, degree, knots):
        super(VCNet_module, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)

        self.Q = nn.Sequential(*blocks)

    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        #t_hidden = torch.cat((torch.unsqueeze(t, 1), x), 1)
        g = self.density_estimator_head(t, hidden)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                
                
# Targeted Regularizer

class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis
        self.weight = nn.Parameter(torch.rand(self.d), requires_grad=True)

    def forward(self, t):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        #self.weight.data.normal_(0, 0.01)
        self.weight.data.zero_()
        
############
# Training #
############

def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()

