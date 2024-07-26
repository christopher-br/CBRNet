# LOAD MODULES
# Standard library
from typing import Tuple, Union, List

# Third party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lightning.pytorch as pl
from lightning.pytorch import Trainer

# Inheritance classes
class TorchDataset(Dataset):
    """
    A PyTorch Dataset class for handling data loading.

    The TorchDataset class is a subclass of PyTorch's Dataset class. It is used to load and preprocess 
    data that will be fed into a PyTorch model. The class should implement the `__len__` method to return 
    the number of items in the dataset and the `__getitem__` method to return the item at a given index.

    Attributes:
        x (torch.Tensor): The covariates. A 2D tensor where each row is an observation 
                          and each column is a covariate.
        y (torch.Tensor): The outcome. A 2D tensor where each element is the outcome for 
                          the corresponding observation.
        d (torch.Tensor): The doses. A 2D tensor where each element is the dose 
                          for the corresponding observation.
        t (torch.Tensor): The treatments. A 2D tensor where each element is the treatment 
                          for the corresponding observation.
        length (int): The number of items in the dataset.

    Methods:
        get_data(self): Returns the covariates, outcome, and treatment as a tuple.
        __len__(self): Returns the number of items in the dataset.
        __getitem__(self, index): Returns the item at a given index.
    """
    def __init__(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        d: np.ndarray, 
        t: np.ndarray
    ) -> None:
        """
        Initializes the TorchDataset object with the provided data.

        This method takes as input the covariates `x`, the outcome `y`, the dose`d`, and the treatment `t`, 
        and initializes the TorchDataset object with this data. The method does not return anything 
        as the data is stored internally in the object.

        Parameters:
            x (np.ndarray): The covariates. A 2D numpy array where each row is an observation 
                            and each column is a covariate.
            y (np.ndarray): The outcome. A 1D numpy array where each element is the outcome for 
                            the corresponding observation.
            d (np.ndarray): The dose. A 1D numpy array where each element is the dose 
                            for the corresponding observation.
            t (np.ndarray): The treatments. A 1D numpy array where each element is the treatment 
                            for the corresponding observation.

        Returns:
            None
        """
        # Assign values according to indices
        self.x = torch.from_numpy(x).type(torch.float32)
        self.y = torch.from_numpy(y).type(torch.float32).reshape(-1, 1)
        self.d = torch.from_numpy(d).type(torch.float32).reshape(-1, 1)
        self.t = torch.from_numpy(t).type(torch.float32).reshape(-1, 1)

        # Save length
        self.length = x.shape[0]

    # Define necessary fcts
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the covariates, outcome, and treatment.

        This method returns the covariates `x`, the outcome `y`, and the treatment `t` 
        that are stored in the TorchDataset object. The method returns a tuple where the 
        first element is the covariates, the second element is the outcome, and the third 
        element is the treatment.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple where the first element is 
            the covariates (a 2D numpy array where each row is an observation and each 
            column is a covariate), the second element is the outcome (a 1D numpy array 
            where each element is the outcome for the corresponding observation), and the 
            third element is the dose (a 1D numpy array where each element is the 
            treatment for the corresponding observation).
        """
        return self.x, self.y, self.d, self.t

    def __getitem__(
        self, 
        index: Union[int, List[int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the item at a given index.

        This method takes as input an index, which can be an integer or a list of integers, 
        and returns the covariates, outcome, and treatment for the observations at these indices. 
        The method returns a tuple where the first element is the covariates, the second element 
        is the outcome, and the third element is the treatment.

        Parameters:
            index (Union[int, List[int]]): The index or indices of the observations to return.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple where the first element is 
            the covariates (a 2D numpy array where each row is an observation and each 
            column is a covariate), the second element is the outcome (a 1D numpy array 
            where each element is the outcome for the corresponding observation), and the 
            third element is the dose (a 1D numpy array where each element is the 
            treatment for the corresponding observation).
        """
        return self.x[index], self.y[index], self.d[index], self.t[index]

    def __len__(self):
        """
        Returns the number of items in the dataset.

        This method returns the number of items in the TorchDataset object, which is the number of 
        observations in the data. The method does not take any parameters.

        Returns:
            int: The number of items in the dataset.
        """
        return self.length
    
class ContinuousCATE:
    """
    This class represents a Continuous Conditional Average Treatment Effect (CATE) estimator.

    The ContinuousCATE class is used to estimate the effect of a treatment on an outcome, 
    given a set of covariates. This is useful in observational studies where the treatment 
    is not randomly assigned, but instead, the assignment depends on the covariates.

    The class provides methods to fit the model to data and to predict the treatment effect 
    for new data points.

    This is an abstract class. The fit and predict methods must be implemented in a subclass.
    """
    def fit(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        d: np.ndarray, 
        t: np.ndarray, 
    ) -> None:
        """
        Fits the model to the provided data.

        This method takes as input the covariates `x`, the outcome `y`, and the treatment `t`, 
        and fits the model to this data. The method does not return anything as the fitted model 
        is stored internally in the object.

        Parameters:
            x (np.ndarray): The covariates. A 2D numpy array where each row is an observation 
                            and each column is a covariate.
            y (np.ndarray): The outcome. A 1D numpy array where each element is the outcome for 
                            the corresponding observation.
            d (np.ndarray): The dose. A 1D numpy array where each element is the dose 
                            for the corresponding observation.
            t (np.ndarray): The treatment. A 1D numpy array where each element is the treatment 
                            for the corresponding observation.

        Returns:
            None
        """
        ...

    def predict(
        self, 
        x: np.ndarray, 
        d: np.ndarray,
        t: np.ndarray, 
    ) -> np.ndarray:
        """
        Predicts the outcome for the provided covariates and treatment.

        This method takes as input the covariates `x` and the treatment `t`, 
        and predicts the outcome based on the model fitted by the `fit` method. 
        The method returns a numpy array where each element is the predicted outcome 
        for the corresponding observation.

        Parameters:
            x (np.ndarray): The covariates. A 2D numpy array where each row is an observation 
                            and each column is a covariate.
            d (np.ndarray): The dose. A 1D numpy array where each element is the dose 
                            for the corresponding observation.
            t (np.ndarray): The treatment. A 1D numpy array where each element is the treatment 
                            for the corresponding observation.

        Returns:
            np.ndarray: The predicted outcomes. A 1D numpy array where each element is the 
                        predicted outcome for the corresponding observation.
        """
        ...

    def score(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        d: np.ndarray, 
        t: np.ndarray,
    ) -> float:
        """
        Computes the score of the model on the provided data.

        This method takes as input the covariates `x`, the outcome `y`, and the treatment `t`, 
        and computes a score that measures how well the model's predictions match the actual outcomes. 
        The method returns a single float that represents this score.

        Parameters:
            x (np.ndarray): The covariates. A 2D numpy array where each row is an observation 
                            and each column is a covariate.
            y (np.ndarray): The actual outcomes. A 1D numpy array where each element is the outcome 
                            for the corresponding observation.
            d (np.ndarray): The dose. A 1D numpy array where each element is the dose 
                            for the corresponding observation.
            t (np.ndarray): The treatment. A 1D numpy array where each element is the treatment 
                            for the corresponding observation.

        Returns:
            float: The score of the model on the provided data. Higher values indicate better fit.
        """
        y_hat = self.predict(x, d, t)

        mse = ((y - y_hat) ** 2).mean()

        return mse

class ContinuousCATENN(pl.LightningModule):
    """
    A PyTorch Lightning Module for CATE estimation for continuous-valued treatments (ContinuousCATENN).

    The ContinuousCATENN class is a subclass of PyTorch Lightning's LightningModule. It represents 
    a neural network designed for causal inference in continuous treatment settings.

    This is an abstract class. Model specific implementations must be implemented in a subclass.
    """
    def __init__(
        self,
        input_size: int,
        num_steps: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        binary_outcome: bool = False,
        hidden_size: Union[int, float] = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
        accelerator: str = "cpu",
    ) -> None:
        """
        Initializes the ContinuousCATENN object with the provided parameters.

        This method takes as input various parameters needed for the ContinuousCATENN and 
        initializes the object with these parameters. The method does not return anything 
        as the parameters are stored internally in the object.

        Parameters:
            input_size (int): The number of input features.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            batch_size (int, optional): The size of the batches for training. Defaults to 64.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int or float, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
            accelerator (str, optional): The device to use for computations. Defaults to "cpu".

        Returns:
            None
        """
        # Ini the super module
        super(ContinuousCATENN, self).__init__()
        # Set seed
        torch.manual_seed(42)

        # Save vars
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.regularization_l2 = regularization_l2
        # Save hidden_size or calc if float is passed
        if type(hidden_size) == int:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = int(hidden_size * input_size)
        self.binary_outcome = binary_outcome
        self.verbose = verbose
        self.activation = activation

        # Initialize trainer
        self.trainer = Trainer(
            max_steps=self.num_steps,
            max_epochs=9999,  # To not interupt step-wise iteration
            fast_dev_run=False,
            reload_dataloaders_every_n_epochs=False,
            enable_progress_bar=self.verbose,
            enable_checkpointing=False,
            enable_model_summary=self.verbose,
            logger=False,
            accelerator=accelerator,
            devices=1,
        )

    def configure_optimizers(self):
        """
        Configures the optimizers for the ContinuousCATENN.

        This method sets up the optimizers to be used during the training of the ContinuousCATENN. 
        The method does not take any parameters and does not return anything as the optimizers are 
        stored internally in the object.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularization_l2,
        )

    def dataloader(
        self, 
        torchDataset: TorchDataset, 
        shuffle: bool = True
    ) -> DataLoader:
        """
        Creates a DataLoader from a TorchDataset.

        This method takes as input a TorchDataset and a boolean indicating whether to shuffle the data, 
        and returns a DataLoader that can be used to iterate over the data in batches.

        Parameters:
            torchDataset (TorchDataset): The dataset to load.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Returns:
            DataLoader: A DataLoader for the given TorchDataset.
        """
        return DataLoader(torchDataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        d: np.ndarray,
        t: np.ndarray,
    ) -> None:
        """
        Fits the ContinuousCATENN to the provided data.

        This method takes as input the covariates `x`, the outcome `y`, and the treatment `t`, 
        and fits the ContinuousCATENN to this data. The method does not return anything as the 
        fitted model is stored internally in the object.

        Parameters:
            x (np.ndarray): The covariates. A 2D numpy array where each row is an observation 
                            and each column is a covariate.
            y (np.ndarray): The outcome. A 1D numpy array where each element is the outcome for 
                            the corresponding observation.
            d (np.ndarray): The dose. A 1D numpy array where each element is the dose 
                            for the corresponding observation.
            t (np.ndarray): The treatment. A 1D numpy array where each element is the treatment 
                            for the corresponding observation.

        Returns:
            None
        """
        t_data = TorchDataset(x, y, d, t)
        loader = self.dataloader(t_data)
        self.trainer.fit(self, loader)

    def score(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        d: np.ndarray,
        t: np.ndarray,
    ) -> float:
        """
        Scores the ContinuousCATENN on the provided data.

        This method takes as input the covariates `x`, the outcome `y`, and the treatment `t`, 
        and returns a score indicating how well the ContinuousCATENN fits this data.

        Parameters:
            x (np.ndarray): The covariates. A 2D numpy array where each row is an observation 
                            and each column is a covariate.
            y (np.ndarray): The outcome. A 1D numpy array where each element is the outcome for 
                            the corresponding observation.
            d (np.ndarray): The dose. A 1D numpy array where each element is the dose 
                            for the corresponding observation.
            t (np.ndarray): The treatment. A 1D numpy array where each element is the treatment 
                            for the corresponding observation.

        Returns:
            float: The score of the ContinuousCATENN on the provided data.
        """
        y_hat = self.predict(x, d, t)

        mse = ((y - y_hat) ** 2).mean()

        return mse

    def predict(
        self, 
        x: np.ndarray, 
        d: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts the outcome for the provided covariates and treatment.

        This method takes as input the covariates `x` and the treatment `t`, and returns a numpy array 
        of predicted outcomes.

        Parameters:
            x (np.ndarray): The covariates. A 2D numpy array where each row is an observation 
                            and each column is a covariate.
            d (np.ndarray): The dose. A 1D numpy array where each element is the dose 
                            for the corresponding observation.
            t (np.ndarray): The treatment. A 1D numpy array where each element is the treatment 
                            for the corresponding observation.

        Returns:
            np.ndarray: The predicted outcomes. A 1D numpy array where each element is the predicted 
                        outcome for the corresponding observation.
        """
        x = torch.tensor(x, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).reshape(-1, 1)
        t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

        y_hat = self.forward(x, d, t).reshape(-1).detach().numpy()

        return y_hat

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        batch_idx: int,
    ) -> float:
        """
        Performs a training step for the ContinuousCATENN.

        This method takes as input a batch of data and the index of the batch, and performs a training 
        step for the ContinuousCATENN. The method returns a dictionary with the loss and any additional 
        metrics to be logged.

        Parameters:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The batch of data. A tuple where 
                the first element is the covariates (a 2D tensor where each row is an observation and 
                each column is a covariate), the second element is the outcome (a 1D tensor where each 
                element is the outcome for the corresponding observation), and the third element is the 
                treatment (a 1D tensor where each element is the treatment for the corresponding observation).
            batch_idx (int): The index of the batch.

        Returns:
            float: The loss for the batch.
        """
        x, y, d, t = batch

        y_hat = self(x, d, t)

        loss_mse = F.mse_loss(y, y_hat)

        return loss_mse

    def forward(
        self,
        x: torch.Tensor, 
        d: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the ContinuousCATENN.

        This method takes as input the covariates `x` and the treatment `t`, and performs a forward 
        pass through the ContinuousCATENN. The method returns a tensor of predicted outcomes.

        Parameters:
            x (torch.Tensor): The covariates. A 2D tensor where each row is an observation and 
                            each column is a covariate.
            d (torch.Tensor): The dose. A 1D tensor where each element is the dose for 
                            the corresponding observation.
            t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

        Returns:
            torch.Tensor: The predicted outcomes. A 1D tensor where each element is the predicted 
                        outcome for the corresponding observation.
        """
        ...