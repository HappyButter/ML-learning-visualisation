from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 1400


def calc_distance_between_nodes(layers: list) -> Tuple[int, list]:
    """Function calculating distances on x and y axis between nodes

    Parameters
    ----------
    layers: List
        A list containing the amount of nodes in each layer of the neural network

    Returns
    -------
    Tuple[int, list]
        Distance on the x axis between the layers and list of distances between nodes on the y axis in each layer

    """
    amount_of_layers = len(layers) + 2
    dist_x = WINDOW_WIDTH // amount_of_layers
    dist_y = []
    for layer in layers:
        dist_y.append(WINDOW_HEIGHT // (layer + 2))
    return dist_x, dist_y


def create_node_position_list(layers: list, dist_x: int, dist_y: list) -> Tuple[list, list]:
    """Function calculating x and y coordinates of every node in the neural network

    Parameters
    ----------
    layers: List
        A list containing the amount of nodes in each layer of the neural network
    dist_x: int
        A int value representing the distance on x axis between each layer
    dist_y: list
        A tuple representing the distance on y axis between nodes in every layer

    Returns
    -------
    Tuple[list, list]
        Tuple containing two lists: One for the x axis positions of every node, and one for the y axis positions

    """
    node_position_list_x = []
    distance_counter = 0
    for layer in layers:
        distance_counter += dist_x
        node_position_list_x.extend([distance_counter for _ in range(layer)])

    node_position_list_y = []
    for number_of_layer, layer_size in enumerate(layers):
        node_position_list_y.extend([dist_y[number_of_layer] * (x + 1) for x in range(layer_size)])

    return node_position_list_x, node_position_list_y


def create_edge_position_list(layers: list, node_position_list_x: list, node_position_list_y: list) -> list:
    """Function calculating x and y coordinates of every edge in the neural network

    Parameters
    ----------
    layers: List
        A list containing the amount of nodes in each layer of the neural network
    node_position_list_x: list
        A list containing x axis positions of every node in the network
    node_position_list_y: list
        A list of lists containing y axis positions of every node in the network

    Returns
    -------
        Position info about every node in the graph

    """
    edge_positions = []

    current_pos = layers[0]

    for layer, next_layer in zip(layers[:-1], layers[1:]):

        for j in range(next_layer):

            x1 = node_position_list_x[current_pos + j]
            y1 = node_position_list_y[current_pos + j]

            for k in range(layer):
                x2 = node_position_list_x[current_pos - layer + k]
                y2 = node_position_list_y[current_pos - layer + k]
                edge_positions.append([(x1, x2), (y1, y2)])

        current_pos += next_layer

    return edge_positions


def create_bias_list(layers: list, bias: list) -> list:
    """Function for creating a list of biases

    Parameters
    ----------
    layers: list
        A list specifying the number of nodes in each layer
    bias: list
        A list of biases

    Returns
    -------
        A list of biases for every node

    """

    bias_list = [0 for _ in range(layers[0])]
    for layer_bias in bias:
        bias_list.extend(layer_bias)

    return bias_list


def create_weights_list(weights: list):
    """Function for creating a list of weights more suited for our app

    Parameters
    ----------
    weights: list
        List of weights

    Returns
    -------
        A list of weights more suited for our rendering function

    """
    weights_list = []
    for weights_layer in weights:
        for weights_from_node in weights_layer:
            weights_list.extend(weights_from_node)
    return weights_list


# weight / 128 -> should be / 2, but we are dividing by larger number to lose float information
def get_rgb_from_weight(weight: float) -> str:
    """Function calculating rgb value from a weight float value

    Parameters
    ----------
    weight: float
        A weight value of the edge
    Returns
    -------
    str
        RGB value in form of a string

    """
    weight += 1
    weight = weight / 128
    weight = 255 - int(weight * 255) * 64 - 40

    return "rgb({0},{0},{0})".format(weight)


def get_weight_group(min_range: float, max_range: float, weights: list, edge_positions: list, threshold: float) \
        -> Tuple[list, list]:
    """Function getting positions of nodes whose weights are in the (min_range, max_range) range

    Parameters
    ----------
    min_range: float
        Smallest float value of weight the function accepts
    max_range: float
        Largest flaot value of weight the function accepts
    weights: list
        List of weights
    edge_positions: list
        List of all edges' positions
    threshold: float
        Minimum absolute value for the weights
    Returns
    -------
    str
        RGB value in form of a string

    """
    edge_pos_x = []
    edge_pos_y = []
    for index, weight in enumerate(weights):
        if min_range < weight < max_range and abs(weight) >= threshold:
            edge_pos_x.extend([*edge_positions[index][0], None])
            edge_pos_y.extend([*edge_positions[index][1], None])
    return edge_pos_x, edge_pos_y


def get_min_weight(weights: list) -> float:
    """Function for returning minimum weight from list of weights

    Parameters
    ----------
    weights: list
        List of weights

    Returns
    -------
        Minimum weight from list

    """
    minimum = 1
    for weight in weights:
        minimum = min(minimum, np.amin(weight))
    return minimum


def get_max_weight(weights: list) -> float:
    """Function for returning minimum weight from list of weights

    Parameters
    ----------
    weights: list
        List of weights

    Returns
    -------
        Maximum weight from list

    """
    maximum = -1
    for weight in weights:
        maximum = max(maximum, np.amax(weight))
    return maximum


def validate_epoch_number_to_draw(epoch_number: int, epoch_number_to_draw: int, max_number: int = 50) -> int:
    """Function validating epochs number that we want to draw

    Parameters
    ----------
    epoch_number: int
        Int value that specifies the amount of epochs our network will go through

    epoch_number_to_draw: int
        Int value that specifies the epochs number we want to draw

    max_number: int
        The maximum amount of epochs we want to render

    Returns
    -------
        Epoch number within our boundaries

    """
    if epoch_number_to_draw < 1:
        return max_number if epoch_number > max_number else epoch_number
    if epoch_number_to_draw > max_number:
        print("max number of epochs to be drawn is {0}".format(max_number))
        return max_number if epoch_number_to_draw > max_number else epoch_number_to_draw
    return epoch_number_to_draw


def visualise_ML(layers: list, bias: list, weights: list, epoch_number: int, epoch_number_to_draw: int = -1,
                 threshold: float = 0.05):
    """Our main function for visualizing the neural network learning process

    Parameters
    ----------
    layers: list
        List containing the amount of nodes in each layer
    bias: list
        List of biases
    weights: list
        List of weights
    epoch_number: int
        Amount of epochs for the NN to process
    epoch_number_to_draw: int
        Amount of epochs we want to draw
    threshold: float
        Minimum absolute value for the weights
    """
    epoch_number_to_draw = validate_epoch_number_to_draw(epoch_number, epoch_number_to_draw, max_number=100)

    # Create figure
    fig = go.Figure(layout={"width": WINDOW_WIDTH, "height": WINDOW_HEIGHT})

    dist_x, dist_y = calc_distance_between_nodes(layers)

    node_position_list_x, node_position_list_y = create_node_position_list(layers, dist_x, dist_y)

    edge_positions = create_edge_position_list(layers, node_position_list_x, node_position_list_y)
    groups = 11

    epochs_to_draw_index_list = np.linspace(0, epoch_number-1, epoch_number_to_draw, dtype=int)

    for epoch in epochs_to_draw_index_list:

        min_weight = get_min_weight(weights[epoch])
        max_weight = get_max_weight(weights[epoch])

        groups_ranges = np.linspace(min_weight, max_weight, num=groups)
        node_colors = create_bias_list(layers, bias[epoch])
        weights_per_epoch = create_weights_list(weights[epoch])

        for index, group_range in enumerate(groups_ranges[:-1]):
            edge_pos_x, edge_pos_y = get_weight_group(group_range,
                                                      groups_ranges[index + 1],
                                                      weights_per_epoch,
                                                      edge_positions,
                                                      threshold)

            fig.add_trace(go.Scattergl(
                visible=False,
                x=edge_pos_x, y=edge_pos_y,
                line=dict(
                    color=get_rgb_from_weight(group_range)
                ),
                hoverinfo='none',
                mode='lines',
            ))

        fig.add_trace(go.Scattergl(
            visible=False,
            x=node_position_list_x, y=node_position_list_y,
            mode='markers',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=node_colors,
                size=25,
                line_width=1)))
    scatter_number_per_epoch = groups

    for i in range(scatter_number_per_epoch):
        fig.data[i].visible = True

    steps = []
    for i in np.arange(0, len(fig.data), scatter_number_per_epoch):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Epoch number: " + str(epochs_to_draw_index_list[i // scatter_number_per_epoch])}],
        )

        for j in range(scatter_number_per_epoch):
            step["args"][0]["visible"][i + j] = True

        steps.append(step)

    sliders = [dict(
        active=0,
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    app.run_server(debug=True, use_reloader=False)
