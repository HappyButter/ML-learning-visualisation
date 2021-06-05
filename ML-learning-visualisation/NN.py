from typing import List, Tuple

import torch
import torch.nn
import torch.nn.functional


class NN:
    """
    Attributes
    ----------
    model: torch.nn.Module
        Neural network model created by PyTorch
    criterion: torch.nn.CrossEntropyLoss
        instance of the criterion
    optimizer: torch.optim.SGD
         instance of the optimizer
    loss_vector: list
        list of loss function value for each epoch
    weights: List[list]
        list of weights for each epoch
    bias: List[list]
        list of bias for each epoch
    layers: list[int]
        List of how many neutrons each layer of model contains

    Parameters
    ----------
    n_in : int
        number of input nodes
    n_hidden1 : int
        number of hidden nodes in first layer
    n_out : int
        number of output nodes
    epoch_number : int
        number of epoch, will be used in training process
    learning_rate : float
        learning rate for optimizer

    """
    def __init__(self, n_in: int, n_hidden1: int, n_out: int, epoch_number: int, learning_rate: float):
        self.epoch_number = epoch_number
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden1, n_out),
            torch.nn.Softmax(dim=1)
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.loss_vector = []
        self.weights = []
        self.bias = []

        self.layers: List[int] = get_layers(self.model)

    def train(self, train_X, train_y):

        for epoch in range(self.epoch_number):
            self.optimizer.zero_grad()
            out = self.model(train_X)
            loss = self.criterion(out, train_y)
            loss.backward()
            weights, bias = extract_weight_bias_from_model(self.model)
            self.weights.append(weights)
            self.bias.append(bias)
            self.optimizer.step()

            if epoch % 100 == 0:
                print('number of epoch', epoch, 'loss', loss.item())

            self.loss_vector.append(loss.item())


def extract_weight_bias_from_model(model: torch.nn.Module) -> Tuple[list, list]:
    """ Extract weights and bias from given neural network model

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model created by PyTorch
    Returns
    -------
    Tuple[list, list]
        Tuple contains list of weights and list of bias of neural network model
    """
    weights = []
    bias = []

    for name, param in model.state_dict().items():
        p = param.detach().numpy().copy()
        if 'weight' in name:
            weights.append(p)
        if 'bias' in name:
            bias.append(p)
    return weights, bias


def get_layers(model: torch.nn.Module) -> List[int]:
    """ Construct layers of neural network based on given model

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model created by PyTorch

    Returns
    -------
    List[int]
        List of how many neutrons each layer of model contains
    """
    layers = []
    parameters = list(model.named_parameters())
    for index, (name, param) in enumerate(parameters):
        if 'weight' in name:
            param = param.detach().numpy()
            layers.append(param.shape[1])
            if index//2 == len(parameters)//2 - 1:
                layers.append(param.shape[0])
    return layers
