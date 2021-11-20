import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        BaseLayer.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.rand(input_size+1, output_size)
        self.read = False
        self._gradient_weights = None
        self.batch_size = None
    def forward(self, input_tensor):
        self.batch_size = np.size(input_tensor,0)
        self.input_tensor = input_tensor
        bias = np.ones((self.batch_size,1))
        self.input_tensor =np.concatenate((self.input_tensor,bias),axis=1)
        tensor = np.dot(self.input_tensor, self.weights)
        return tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter                       #setter must behind the getter of that property
    def optimizer(self, value):
        self._optimizer = value
        self.read = True

    def backward(self, error_tensor):
        self.error_tensor = error_tensor

        self.gradient_weights = np.dot( self.input_tensor.T , self.error_tensor)

        if self.read is True:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        weights = self.weights.T[:, :-1]
        tensor = np.dot(error_tensor, weights)

        return tensor

    @property
    def gradient_weights(self):

        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):

        self._gradient_weights = value
if __name__ == "__main__":
    layer = FullyConnected(4, 3)
    layer.optimizer = 1
    print(layer.optimizer)