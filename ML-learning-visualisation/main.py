import copy
import torch
from torch.autograd import Variable
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from visualization import *
import numpy as np

from NN import NN

def visualise_ML(layers, bias, weights):
    epoch_number = len(bias)
    print(epoch_number)
    edge_colors = []
    
    # Create figure
    fig = go.Figure()
    
    dist_x, dist_y = calc_distance_between_nodes(layers)

    node_position_list_x, node_position_list_y = create_node_position_list(layers, dist_x, dist_y)

    edge_position_list_x, edge_position_list_y = create_edge_position_list(layers, node_position_list_x, node_position_list_y)
    
    for i in range(epoch_number):
        node_colors = create_bias_list(layers, bias[i])
        edge_colors = create_weights_list(layers, weights[i])
        

        for j in range(len(edge_colors)):
            fig.add_trace(go.Scatter(
                visible=False,
                x=edge_position_list_x[3*j:3*j+3], y=edge_position_list_y[3*j:3*j+3],
                line=dict(
                    color=get_rgb_from_weight(edge_colors[j])
                ),
                hoverinfo='none',
                mode='lines',
            ))


        fig.add_trace(go.Scatter(
            visible=False,
            x=node_position_list_x, y=node_position_list_y,
            mode='markers',
            marker=dict(
                colorscale='YlGnBu',
                reversescale=True,
                color=node_colors,
                size=25,
                line_width=1)))

    scatter_number_per_epoch = len(edge_colors) + 1
    
    for i in range(scatter_number_per_epoch):
        fig.data[i].visible = True
    
    
    # Create and add slider
    steps = []
    for i in np.arange(0, len(fig.data), scatter_number_per_epoch):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}], 
        )
        
        for j in range(scatter_number_per_epoch):
            step["args"][0]["visible"][i+j] = True
    
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "epoch: "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        showlegend=False
    )
    
    fig.show()


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
    print("confusion_matrix")
    print(confusion_matrix(test_y.data, predict_y.data))
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    print(classification_report(test_y.data, predict_y.data, target_names=labels))

    node_colors = create_bias_list(net.layers, net.bias[0])

    edge_colors = create_weights_list(net.layers, net.weights[0])

    visualise_ML(net.layers, net.bias, net.weights)



if __name__ == '__main__':
    main()