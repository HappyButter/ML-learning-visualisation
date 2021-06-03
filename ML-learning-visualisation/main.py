import torch
from torch.autograd import Variable

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


from visualization import visualise_ML
from NN import NN


def main():
    # load iris dataset
    iris = load_iris()

    # create test/train data
    datasets = train_test_split(iris.data, iris.target, test_size=0.2)
    train_X, test_X, train_y, test_y = datasets
    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).long())
    test_y = Variable(torch.Tensor(test_y).long())

    # construct and train network
    net = NN()
    net.train(train_X, train_y)

    # test network
    predict_out = net.model(test_X)
    _, predict_y = torch.max(predict_out, 1)
    # print("confusion_matrix")
    # print(confusion_matrix(test_y.data, predict_y.data))
    # labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    # print(classification_report(test_y.data, predict_y.data, target_names=labels))


    visualise_ML(net.layers, net.bias, net.weights, 1)



if __name__ == '__main__':
    main()