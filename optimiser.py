from CherialNetwork.nn import NeuralNetwork

class Optimizer:
    def step(self , net:NeuralNetwork)->None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self,learning_rate : float=0.01)-> None:
        self.learning_rate = learning_rate

    def step(self , net:NeuralNetwork)->None:
        for param,grad in net.params_and_grads():
            param -= self.learning_rate * grad
