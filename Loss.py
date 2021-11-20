from Layers.Base import BaseLayer
import numpy as np


class CrossEntropyLoss():
    def __init__(self):
        BaseLayer.__init__(self)
    def forward(self,prediction_tensor,label_tensor):

        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        eps = np.finfo(float).eps
        logit = - np.log(self.prediction_tensor + eps)

        tensor = np.sum(logit[label_tensor == 1])

        return tensor

    def backward(self,label_tensor):
        self.label_tensor = label_tensor
        error_tensor = - np.divide(self.label_tensor, self.prediction_tensor)
        return error_tensor
