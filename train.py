from CherialNetwork.nn import NeuralNetwork
from CherialNetwork.tensor import Tensor
from CherialNetwork.cost_function import Loss,MeanSquaredError
from CherialNetwork.data import DataIterator,BatchIterator
from CherialNetwork.optimiser import SGD, Optimizer

def train(net: NeuralNetwork,
            inputs: Tensor,
            targets: Tensor,
            num_epochs: int = 5000,
            iterator: DataIterator = BatchIterator(),
            loss: Loss=MeanSquaredError(),
            optimizer: Optimizer=SGD())->None:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in iterator(inputs,targets):
                predicted = net.forward(batch.inputs)
                epoch_loss += loss.loss(predicted,batch.targets)
                grad = loss.grad(predicted,batch.targets)
                net.backward(grad)
                optimizer.step(net)
            print(epoch,epoch_loss )
                 
            