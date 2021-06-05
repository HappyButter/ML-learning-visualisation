import torch
import torch.nn
import torch.nn.functional


class NN:
    def __init__(self):

        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20),
            torch.nn.Linear(20, 3),
            torch.nn.Softmax(dim=1)
        )
        self.loss_vector = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.weights = []
        self.bias = []
        self.layers = get_layers(self.model)

        weights, bias = extract_weight_bias(self.model)
        self.weights.append(weights)
        self.bias.append(bias)

    def train(self, train_X, train_y):

        for epoch in range(1000):
            self.optimizer.zero_grad()
            out = self.model(train_X)
            loss = self.criterion(out, train_y)
            loss.backward()
            weights, bias = extract_weight_bias(self.model)
            self.weights.append(weights)
            self.bias.append(bias)
            self.optimizer.step()

            # if epoch % 100 == 0:
            #     print('number of epoch', epoch, 'loss', loss.item())

            self.loss_vector.append(loss.item())


def extract_weight_bias(model):
    weights = []
    bias = []

    for name, param in model.state_dict().items():
        p = param.detach().numpy().copy()
        if 'weight' in name:
            weights.append(p)
        if 'bias' in name:
            bias.append(p)
    return weights, bias


def get_layers(model):
    layers = []
    for name, param in model.named_parameters():
        if 'weight' in name and len(layers) == 0:
            param = param.detach().numpy()
            layers.append(param.shape[1])
        if 'bias' in name:
            param = param.detach().numpy()

            layers.append(param.shape[0])
    return layers

