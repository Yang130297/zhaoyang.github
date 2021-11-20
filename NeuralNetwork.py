from Layers.Base import BaseLayer
import numpy as np
from copy import deepcopy


class NeuralNetwork(BaseLayer):
    def __init__(self, optimizer):
        BaseLayer.__init__(self)
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.loss_layer = None
        self.data_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            # input_tensor ---->every layer---->forward---->input_tensor ---->next layer
            self.input_tensor = layer.forward(self.input_tensor)
        # But loss.forward(prediction_tensor, label_tensor) need two argument
        output = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        return output

    def backward(self):
        tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            # input_tensor ---->every layer---->forward---->input_tensor ---->next layer
            tensor = layer.backward(tensor)

        # loss.backward got prediction_tensor in same layer
        # we just give it the label_tensor
        return tensor

    def append_layer(self, layer):

        if layer.trainable is True:
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        # Loss calculation is completely in Loss
        self.loss = np.zeros(iterations)
        for loss_iterations in range(iterations):
            self.loss[loss_iterations] = self.forward()
            # Only the right-label will be marked as 1, the others is not important
            self.backward()

    def test(self, input_tensor):

        # propagates the input_tensor through the network ---> forward(input_tensor)
        prob_output = input_tensor
        for layer in self.layers:  # ---> every layer
            prob_output = layer.forward(prob_output)  # input_tensor ---> input_tensor of every layer
        return prob_output
        # output_tensor of Softmax(Softmax.Softmax.forward(input_tensor))
