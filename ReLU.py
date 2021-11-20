from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)

    def forward(self,input_tensor):

        self.input_tensor = input_tensor

        tensor = input_tensor

        tensor[tensor <= 0] = 0

        return tensor

    def backward(self,error_tensor):

        error_tensor[self.input_tensor <= 0] = 0

        return error_tensor
