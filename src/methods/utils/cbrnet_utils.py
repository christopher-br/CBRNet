# LOAD MODULES
# Standard library
...

# Third party
import torch
from torch import nn
import ot
from sklearn.cluster import KMeans

class Clusterer:
    """
    This is an inheritance class for all clusterers.
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initializes a new instance of the class.
        """
        ...
    
    def fit(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ):
        """
        Fits the clusterer to the data.
        """
        ...
        
    def predict(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ):
        """
        Predicts the clusters for the data.
        """
        ...

class KMeansClusterer(Clusterer):
    """
    KMeans clusterer.
    """
    def __init__(
        self,
        n_clusters: int,
        **kwargs,
    ):
        """
        Initializes a new instance of the class.
        """
        self.n_clusters = n_clusters
    
    def fit(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ):
        """
        Fits the clusterer to the data.
        """
        xdt = torch.cat((x, d, t), dim=1)
        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            random_state=42,
            n_init="auto",
        ).fit(xdt)
        
    def predict(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ):
        """
        Predicts the clusters for the data.
        """
        xdt = torch.cat((x, d, t), dim=1)
        cluster = self.clusterer.predict(xdt)
        cluster = torch.tensor(cluster)
        
        return cluster
    
class MMD(nn.Module):
    # Inspired by https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
    def __init__(
        self,
        kernel: str="linear"
    ):
        super().__init__()
        
        # Define kernels
        class Lin(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, X):
                return torch.mm(X,X.T)
        
        class RBF(nn.Module):
            def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
                super().__init__()
                self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
                self.bandwidth = bandwidth

            def get_bandwidth(self, L2_distances):
                if self.bandwidth is None:
                    n_samples = L2_distances.shape[0]
                    return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

                return self.bandwidth

            def forward(self, X):
                L2_distances = torch.cdist(X, X) ** 2
                return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)
        
        # Save kernel
        if kernel not in ["linear", "rbf"]:
            raise ValueError("Kernel must be either 'linear' or 'rbf'.")
        
        self.Kernel = Lin() if (kernel == "linear") else RBF()
    
    def forward(self, sample_0, sample_1):
        """
        Compute the MMD for a specified kernel.
        """
        size_0 = sample_0.shape[0]
        K = self.Kernel(torch.vstack([sample_0,sample_1]))
        
        XX = K[:size_0, :size_0].mean()
        XY = K[:size_0, size_0:].mean()
        YY = K[size_0:, size_0:].mean()
        
        mmd_dist = XX - 2 * XY + YY
        
        return mmd_dist

class Wasserstein(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, sample_0, sample_1):
        """
        Compute the Wasserstein distance between two samples.
        """
        # Catch error when sample size too small
        if sample_0.shape[0] == 0 or sample_1.shape[0] == 0:
            return torch.tensor(0.0)
        
        M = ot.dist(sample_0,sample_1)
        
        w0 = torch.ones(sample_0.shape[0])
        w1 = torch.ones(sample_1.shape[0])
        
        wass_dist = ot.emd2(w0 / (w0.sum()), w1 / (w1.sum()), M)
        
        return wass_dist