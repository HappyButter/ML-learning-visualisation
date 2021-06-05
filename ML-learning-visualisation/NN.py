from typing import Tuple

import torch
import torch.nn
import torch.nn.functional


class NN:
    def __init__(self, n_in: int, n_hidden1: int, n_out: int, epoch_number: int, learning_rate: float):

        self.epoch_number = epoch_number
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden1, n_out),
            torch.nn.Softmax(dim=1)
        )
        self.loss_vector = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.weights = []
        self.bias = []
        self.layers: list = get_layers(self.model)

        weights, bias = extract_weight_bias_from_model(self.model)
        self.weights.append(weights)
        self.bias.append(bias)

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


def extract_weight_bias_from_model(model) -> Tuple[list, list]:
    weights = []
    bias = []

    for name, param in model.state_dict().items():
        p = param.detach().numpy().copy()
        if 'weight' in name:
            weights.append(p)
        if 'bias' in name:
            bias.append(p)
    return weights, bias


def get_layers(model) -> list:
    layers = []
    parameters = list(model.named_parameters())
    for index, (name, param) in enumerate(parameters):
        if 'weight' in name:
            param = param.detach().numpy()
            layers.append(param.shape[1])
            if index//2 == len(parameters)//2 - 1:
                layers.append(param.shape[0])
    return layers
