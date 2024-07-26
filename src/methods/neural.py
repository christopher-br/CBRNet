# LOAD MODULES

# Standard library
from typing import Callable
import math

# Proprietary
from src.methods.utils.classes import (
    # Standard imports
    TorchDataset,
    ContinuousCATENN,
    ContinuousCATE,
)

from src.methods.utils.cbrnet_utils import (
    # CBRNet imports
    KMeansClusterer,
    MMD,
    Wasserstein
)

from src.methods.utils.drnet_utils import (
    # DRNet imports
    DRNetHeadLayer,
)

from src.methods.utils.vcnet_utils import (
    # VCNet imports
    VCNet_module,
    TR,
    get_iter,
    criterion,
    criterion_TR,
)

# Third party
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os

class MLP(ContinuousCATENN):
    """
    Multilayer Perceptron (MLP) class that inherits from the ContinuousCATENN class.

    This class represents a Multilayer Perceptron, which is a type of neural network. It inherits 
    from the ContinuousCATENN class.
    """
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_layers: int = 2,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            input_size (int): The size of the input to the network.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            num_layers (int, optional): The number of layers in the network. Defaults to 2.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        # Ini the super module
        super(MLP, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.num_layers = num_layers

        # Structure
        # Shared layers
        self.layers = nn.Sequential(nn.Linear(self.input_size + 2, self.hidden_size)) # +2 for the dose and treatment
        self.layers.append(self.activation)
        # Add additional layers
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(self.activation)
        # Add output layer
        self.layers.append(nn.Linear(self.hidden_size, 1))
        # Sigmoid activation if binary is True
        if self.binary_outcome == True:
            self.layers.append(nn.Sigmoid())

    def forward(
        self, 
        x: torch.Tensor,
        d: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the network using the input tensor `x`, dose tensor `d` and treatment tensor `t`.
        """
        x = torch.cat((x, d, t), dim=1)

        # Feed through layers
        x = self.layers(x)

        return x

class CBRNet(ContinuousCATENN):
    """
    The CBRNet class.
    
    The network clusters observations based on their location in input space and regularizes for distances between clusters in latent space.
    """
    def __init__(
        self,
        input_size: int,
        IPM: Callable = MMD("rbf"),
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_representation_layers: int = 2,
        num_inference_layers: int = 2,
        Clusterer: Callable = KMeansClusterer,
        num_cluster: int = 5,
        regularization_ipm: float = 0.5,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.
        
        Parameters:
            input_size (int): The size of the input to the network.
            IPM (Callable): A callable that computes the IPM.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            num_representation_layers (int, optional): The number of representation layers in the network. Defaults to 2.
            num_inference_layers (int, optional): The number of inference layers in the network. Defaults to 2.
            Clusterer (Callable, optional): A callable that clusters observations. Defaults to kmeans_cluster.
            num_cluster (int, optional): The number of clusters. Defaults to 5.
            regularization_ipm (float, optional): The regularization strength for the Integral Probability Metric. Defaults to 0.5.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        # Ini the super module
        super(CBRNet, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.Clusterer = Clusterer(num_cluster)
        self.num_representation_layers = num_representation_layers
        self.num_inference_layers = num_inference_layers
        self.num_cluster = num_cluster
        self.regularization_ipm = regularization_ipm
        self.IPM = IPM

        # Structure
        # Representation learning layers
        self.representation_layers = nn.Sequential()
        self.representation_layers.append(nn.Linear(self.input_size + 1, self.hidden_size)) # +1 for the treatment
        self.representation_layers.append(self.activation)
        # Add layers
        for i in range(self.num_representation_layers - 1):
            self.representation_layers.append(
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.representation_layers.append(self.activation)

        # Head layers
        self.head_layers = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size) # +1 for the dose
        )
        self.head_layers.append(self.activation)
        # Add additional hidden layers
        for i in range(self.num_inference_layers - 1):
            self.head_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.head_layers.append(self.activation)
        # Add output layer
        self.head_layers.append(nn.Linear(self.hidden_size, 1))
        # Sigmoid activation if binary is True
        if self.binary_outcome == True:
            self.head_layers.append(nn.Sigmoid())

    def fit(self, x, y, d, t):
        """
        Fits the network to the given data.
        
        First, a kmeans clustering is performed on the data. Then, the network is trained using the given data.
        """
        # Build clustering fct
        t_data = TorchDataset(x, y, d, t)
        
        # Train cluster fct
        self.Clusterer.fit(t_data.x, t_data.d, t_data.t)
        
        # Get mode cluster
        cluster = self.Clusterer.predict(t_data.x, t_data.d, t_data.t)
        cluster = torch.tensor(cluster)
        self.modal_cluster = torch.bincount(cluster.flatten()).argmax()
        
        super().fit(x, y, d, t)

    def forward(self, x, d, t):
        """
        Performs a forward pass through the network using the input tensor `x`, dose tensor `d`, and the treatment tensor `t`.
        """
        x = torch.cat((x, t), dim=1)
        hidden = self.representation_layers(x)

        # Add d to x
        x = torch.cat((hidden, d), dim=1)

        # Feed through layers
        x = self.head_layers(x)

        return x, hidden

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step using the given batch of data.
        """
        x, y, d, t = batch
        
        cluster = self.Clusterer.predict(x, d, t)

        y_hat, hidden = self(x, d, t)
        
        # Calculate mse loss
        loss_mse = F.mse_loss(y, y_hat)
        
        # Calculate ipm loss
        loss_ipm = torch.tensor(0).float()
        for c in torch.unique(cluster):
            if c == self.modal_cluster:
                continue
            loss_ipm += self.IPM(hidden[cluster == self.modal_cluster], hidden[cluster == c])
        
        loss_ipm = loss_ipm / (self.num_cluster - 1)

        loss = loss_mse + self.regularization_ipm * loss_ipm
        
        # print(f"mse_loss: {loss_mse.item()}"+f"; ipm_loss: {loss_ipm.item()}")

        return loss

    def predict(self, x, d, t):
        """
        Predicts the outcome for the given data.
        """
        x = torch.tensor(x, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).reshape(-1, 1)
        t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

        y_hat, _ = self.forward(x, d, t)

        y_hat = y_hat.reshape(-1).detach().numpy()

        return y_hat

class DRNet(ContinuousCATENN):
    """
    The DRNet class.
    
    The network uses a binning approach to estimate the outcome.
    """
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_representation_layers: int = 2,
        num_inference_layers: int = 2,
        num_bins: int = 10,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            input_size (int): The size of the input to the network.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            num_representation_layers (int, optional): The number of representation layers in the network. Defaults to 2.
            num_inference_layers (int, optional): The number of inference layers in the network. Defaults to 2.
            num_bins (int, optional): The number of bins for the histogram. Defaults to 10.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        # Ini the super module
        super(DRNet, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.num_representation_layers = num_representation_layers
        self.num_inference_layers = num_inference_layers
        self.num_bins = num_bins

        # Define binning fct
        bounds = torch.linspace(
            0 - torch.finfo().eps, 1 + torch.finfo().eps, (self.num_bins + 1)
        )

        def binning_fct(d: torch.FloatTensor) -> torch.FloatTensor:
            """
            Function that bins observations based on their factual dose.
            """
            # Define bounds
            bins = torch.bucketize(d, bounds) - 1
            return bins

        self.binning_fct = binning_fct

        # Structure
        # Shared layers
        self.shared_layers = nn.Sequential(nn.Linear(self.input_size + 1, self.hidden_size)) # +1 for the treatment
        self.shared_layers.append(self.activation)
        # Add additional layers
        for i in range(self.num_representation_layers - 1):
            self.shared_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.shared_layers.append(self.activation)

        # Head networks
        self.head_networks = nn.ModuleList()
        for i in range(self.num_bins):
            # Build network per head
            help_head = nn.Sequential()
            for j in range(num_inference_layers):
                help_head.append(
                    DRNetHeadLayer(
                        self.hidden_size, self.hidden_size, activation=self.activation
                    )
                )
            # Append last layer
            help_head.append(
                DRNetHeadLayer(
                    self.hidden_size, 1, activation=self.activation, last_layer=True
                )
            )
            if self.binary_outcome == True:
                help_head.append(nn.Sigmoid())
            # Append to module list
            self.head_networks.append(help_head)

    def forward(self, x, d, t):
        """
        Defines the forward pass through the network.
        
        Passes data through the shared layers and then through the head layers.
        Saves the result according to the correct bin.
        """
        x = torch.cat((x, t), dim=1)
        x = self.shared_layers(x)

        # Add d
        hidden = torch.cat((x, d), dim=1)

        # Dump x
        x = torch.zeros((d.shape))

        # Get bins
        bins = self.binning_fct(d)

        # Feed through head layers
        for i in range(self.num_bins):
            head_out = self.head_networks[i](hidden)
            # Set 0, if in wrong head
            x = x + head_out * (bins == i)

        return x
    
class VCNet:
    """
    The VCNet class.
    
    Taken from oringinal implementation. with minor modifications to match style of the rest of the repo.
    """
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        batch_size: int = 500,
        num_steps: int = 1000,
        num_grid: int = 10,
        knots: list = [0.33, 0.66],
        degree: int = 2,
        targeted_regularization: bool = True,
        hidden_size: int = 50,
        wd: float = 5e-3,
        tr_wd: float = 5e-3,
        tr_knots: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        tr_degree: int = 2,
        momentum: float = 0.9,
        init_learning_rate: float = 0.0001,
        alpha: float = 0.5,
        tr_init_learning_rate: float = 0.001,
        beta: float = 1.0,
        binary_outcome: bool = False,
        verbose: bool = False,
    ) -> None:
        # Set seed
        torch.manual_seed(42)

        # Save settings
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.input_size = input_size + 1 # +1 for treatment
        self.learning_rate = learning_rate
        # Save hidden_size or calc if float is passed
        if type(hidden_size) == int:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = int(hidden_size * input_size)
        self.num_grid = num_grid
        self.knots = knots
        self.degree = degree
        self.targeted_regularization = targeted_regularization
        self.wd = wd
        self.tr_wd = tr_wd
        self.tr_knots = tr_knots
        self.tr_degree = tr_degree
        self.momentum = momentum
        self.init_learning_rate = init_learning_rate
        self.alpha = alpha
        self.tr_init_learning_rate = tr_init_learning_rate
        self.beta = beta
        self.binary_outcome = binary_outcome
        self.verbose = verbose

    def fit(self, x, y, d, t):
        # Get num epochs
        num_epochs = math.ceil((self.batch_size * self.num_steps)/x.shape[0])

        train_matrix = torch.from_numpy(np.column_stack((d, x, t, y))).float() # Added treatment

        # Define loader
        train_loader = get_iter(train_matrix, self.batch_size, shuffle=True)

        # Define settings
        cfg_density = [
            (self.input_size, self.hidden_size, 1, "relu"),
            (self.hidden_size, self.hidden_size, 1, "relu"),
        ]
        cfg = [
            (self.hidden_size, self.hidden_size, 1, "relu"),
            (self.hidden_size, 1, 1, "id"),
        ]

        # Load model
        self.model = VCNet_module(
            cfg_density, self.num_grid, cfg, self.degree, self.knots, self.binary_outcome
        )
        self.model._initialize_weights()

        if self.targeted_regularization:
            self.TargetReg = TR(self.tr_degree, self.tr_knots)
            self.TargetReg._initialize_weights()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.init_learning_rate,
            momentum=self.momentum,
            weight_decay=self.wd,
            nesterov=True,
        )

        if self.targeted_regularization:
            tr_optimizer = torch.optim.SGD(
                self.TargetReg.parameters(),
                lr=self.tr_init_learning_rate,
                weight_decay=self.tr_wd,
            )

        # Train
        for epoch in tqdm(
            range(num_epochs),
            leave=False,
            desc="Train VCNet",
            disable=not (self.verbose),
        ):
            for idx, (inputs, y) in enumerate(train_loader):
                idx = idx
                d = inputs[:, 0]
                x = inputs[:, 1:]

                # Train with target reg
                if self.targeted_regularization:
                    optimizer.zero_grad()
                    out = self.model.forward(x, d)
                    trg = self.TargetReg(d)
                    loss = criterion(out, y, alpha=self.alpha) + criterion_TR(
                        out, trg, y, beta=self.beta
                    )
                    loss.backward()
                    optimizer.step()

                    tr_optimizer.zero_grad()
                    out = self.model.forward(x, d)
                    trg = self.TargetReg(d)
                    tr_loss = criterion_TR(out, trg, y, beta=self.beta)
                    tr_loss.backward()
                    tr_optimizer.step()
                # Train withouth target reg
                else:
                    optimizer.zero_grad()
                    out = self.model.forward(x, d)
                    loss = criterion(out, y, alpha=self.alpha)
                    loss.backward()
                    optimizer.step()

    def score(self, x, y, d, t):
        y_hat = self.predict(x, d, t)

        mse = ((y - y_hat) ** 2).mean()

        return mse

    def predict(self, x, d, t):
        pred_matrix = torch.from_numpy(np.column_stack((d, x, t, d))).float()
        # Define pred loader
        pred_loader = get_iter(pred_matrix, pred_matrix.shape[0], shuffle=False)

        for idx, (inputs, y) in enumerate(pred_loader):
            # Get inputs
            d = inputs[:, 0]
            x = inputs[:, 1:]

            # Get estimates
            y_hat = self.model.forward(x, d)[1].data.squeeze().numpy()

        return y_hat
