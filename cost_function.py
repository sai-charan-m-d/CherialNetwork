'''

'''
import numpy as np
from CherialNetwork.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor,actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor,actual: Tensor) -> Tensor:
        raise NotImplementedError

'''
Mean Squared Error
'''
class MeanSquaredError(Loss):
    def loss(self, predicted: Tensor,actual: Tensor) -> float:
        return np.sum((predicted-actual)**2)

    def grad(self, predicted: Tensor,actual: Tensor) -> Tensor:
        return 2*(predicted - actual)

class MeanAbsoluteError(Loss):
    def loss(self, predicted: Tensor,actual: Tensor) -> float:
        return np.sum(np.abs(predicted-actual))

    def grad(self, predicted: Tensor,actual: Tensor) -> Tensor:
        return 2*(predicted - actual)
