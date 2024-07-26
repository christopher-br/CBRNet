# LOAD MODULES
# Standard library
from typing import Callable

# Third party
import numpy as np
from dataclasses import dataclass

# FUNCTIONS
@dataclass
class ContinuousData:
    """
    Creates a dataclass for continuous data.
    """
    x: np.ndarray
    t: np.ndarray
    d: np.ndarray
    y: np.ndarray
    ground_truth: Callable
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    info: str = "No info provided for this dataset."
    
    # ids
    @property
    def train_val_ids(self):
        return np.concatenate((self.train_ids, self.val_ids))
    
    # x
    @property
    def x_train(self):
        return self.x[self.train_ids]
    
    @property
    def x_val(self):
        return self.x[self.val_ids]
    
    @property
    def x_train_val(self):
        return self.x[self.train_val_ids]
    
    @property
    def x_test(self):
        return self.x[self.test_ids]
    
    # t
    @property
    def t_train(self):
        return self.t[self.train_ids]
    
    @property
    def t_val(self):
        return self.t[self.val_ids]
    
    @property
    def t_train_val(self):
        return self.t[self.train_val_ids]
    
    @property
    def t_test(self):
        return self.t[self.test_ids]
    
    # d
    @property
    def d_train(self):
        return self.d[self.train_ids]
    
    @property
    def d_val(self):
        return self.d[self.val_ids]
    
    @property
    def d_train_val(self):
        return self.d[self.train_val_ids]
    
    @property
    def d_test(self):
        return self.d[self.test_ids]
    
    # y
    @property
    def y_train(self):
        return self.y[self.train_ids]
    
    @property
    def y_val(self):
        return self.y[self.val_ids]
    
    @property
    def y_train_val(self):
        return self.y[self.train_val_ids]
    
    @property
    def y_test(self):
        return self.y[self.test_ids]
    
    # xd
    @property
    def xd(self):
        return np.column_stack((self.x, self.d))
    
    @property
    def xd_train(self):
        return np.column_stack((self.x, self.d))[self.train_ids]
    
    @property
    def xd_val(self):
        return np.column_stack((self.x, self.d))[self.val_ids]
    
    @property
    def xd_train_val(self):
        return np.column_stack((self.x, self.d))[self.val_ids][self.train_val_ids]
    
    @property
    def xd_test(self):
        return np.column_stack((self.x, self.d))[self.test_ids]
    
    # xt
    @property
    def xt(self):
        return np.column_stack((self.x, self.t))
    
    @property
    def xt_train(self):
        return self.xt[self.train_ids]
    
    @property
    def xt_val(self):
        return self.xt[self.val_ids]
    
    @property
    def xt_train_val(self):
        return self.xt[self.train_val_ids]
    
    @property
    def xt_test(self):
        return self.xt[self.test_ids]
    
    # xtd
    @property
    def xtd(self):
        return np.column_stack((self.x, self.t, self.d))
    
    @property
    def xtd_train(self):
        return np.column_stack((self.x, self.t, self.d))[self.train_ids]
    
    @property
    def xtd_val(self):
        return np.column_stack((self.x, self.t, self.d))[self.val_ids]
    
    @property
    def xtd_train_val(self):
        return np.column_stack((self.x, self.t, self.d))[self.train_val_ids]
    
    @property
    def xtd_test(self):
        return np.column_stack((self.x, self.t, self.d))[self.test_ids]
    
    def keys(self):
        return [
            'x', 't', 'd', 'y', 
            'ground_truth', 
            'train_ids', 'val_ids', 'test_ids', 
            'info',
            'train_val_ids', 
            'x_train', 'x_val', 'x_train_val', 'x_test',
            't_train', 't_val', 't_train_val', 't_test',
            'd_train', 'd_val', 'd_train_val', 'd_test',
            'y_train', 'y_val', 'y_train_val', 'y_test',
            'xd', 'xd_train', 'xd_val', 'xd_train_val', 'xd_test',
            'xt', 'xt_train', 'xt_val', 'xt_train_val', 'xt_test',
            'xtd', 'xtd_train', 'xtd_val', 'xtd_train_val', 'xtd_test'
        ]