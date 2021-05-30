import torch
from torch.autograd import Variable
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from visualization import *

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
    print("confusion_matrix")
    print(confusion_matrix(test_y.data, predict_y.data))
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    print(classification_report(test_y.data, predict_y.data, target_names=labels))


    layers = net.layers
    dist_x, dist_y = calc_distance_between_nodes(layers)
    node_position_list_x, node_position_list_y = create_node_position_list(layers, dist_x, dist_y)
    edge_position_list_x, edge_position_list_y = create_edge_position_list(layers, node_position_list_x,
                                                                           node_position_list_y)

    edge_trace = go.Scatter(
        x=edge_position_list_x, y=edge_position_list_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_position_list_x, y=node_position_list_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Neural network structure visualization with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()



if __name__ == '__main__':
    main()
