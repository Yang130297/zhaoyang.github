import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_size = np.size(input_tensor,1)
        batch_size = np.size(input_tensor,0)
        self.input_size = input_size
        self.batch_size = batch_size
        X_max = np.max(self.input_tensor,axis=1)
        X_minusmax = self.input_tensor - np.tile(X_max,(input_size,1)).T
        self.output_tensor = np.divide(np.exp(X_minusmax), np.tile(np.sum(np.exp(X_minusmax),axis=1),(input_size,1)).T)
        return self.output_tensor

    def backward(self, error_tensor):
        N = np.size(error_tensor, 1)
        error_dot_output = np.sum(error_tensor * self.output_tensor, axis=1)
        error_tile = np.tile(error_dot_output, (N, 1)).T
        tensor = self.output_tensor * (error_tensor - error_tile)

        return tensor


