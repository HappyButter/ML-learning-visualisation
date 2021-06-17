import torch

from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from visualization import visualise_ML
from NN import NN


def iris_classification():
    iris = load_iris()

    # split data to train and test dataset
    dataset = train_test_split(iris.data, iris.target, test_size=0.2)
    train_X, test_X, train_y, test_y = dataset
    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).long())
    test_y = Variable(torch.Tensor(test_y).long())

    # construct and train network
    net = NN(4, 20, 3, 1000, 0.01)
    net.train(train_X, train_y)

    # test network
    predict_out = net.model(test_X)
    _, predict_y = torch.max(predict_out, 1)
    print("confusion_matrix")
    print(confusion_matrix(test_y.data, predict_y.data))
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    print(classification_report(test_y.data, predict_y.data, target_names=labels))

    # visualize learning process of neutral network
    visualise_ML(layers=net.layers, bias=net.bias, weights=net.weights,
                 epoch_number=len(net.bias), epoch_number_to_draw=1000)


def wine_classification():
    wine = load_wine()

    dataset = train_test_split(wine.data, wine.target, test_size=0.2)
    train_X, test_X, train_y, test_y = dataset

    scale = MinMaxScaler()
    train_X = scale.fit_transform(train_X)
    test_X = scale.fit_transform(test_X)
    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).long())
    test_y = Variable(torch.Tensor(test_y).long())

    # construct and train network
    net = NN(13, 20, 3, 1000, 0.2)
    net.train(train_X, train_y)

    # test network
    predict_out = net.model(test_X)
    _, predict_y = torch.max(predict_out, 1)
    print("confusion_matrix")
    print(confusion_matrix(test_y.data, predict_y.data))
    labels = ['class_0', 'class_1', 'class_2']
    print(classification_report(test_y.data, predict_y.data, target_names=labels))

    # visualize learning process of neutral network
    visualise_ML(layers=net.layers, bias=net.bias, weights=net.weights,
                 epoch_number=len(net.bias), epoch_number_to_draw=1000)


if __name__ == '__main__':
    iris_classification()
    wine_classification()
