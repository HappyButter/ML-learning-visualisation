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
    amount_of_layers = len(layers) + 2
    dist_x = WINDOW_WIDTH // amount_of_layers
    dist_y = []
    for layer in layers:
        dist_y.append(WINDOW_HEIGHT // (layer + 2))
    return dist_x, dist_y


def create_node_position_list(layers: list, dist_x: int, dist_y: list) -> Tuple[list, list]:
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
    bias_list = [0 for _ in range(layers[0])]
    for layer_bias in bias:
        bias_list.extend(layer_bias)

    return bias_list


def create_weights_list(weights: list):
    weights_list = []
    for weights_layer in weights:
        for weights_from_node in weights_layer:
            weights_list.extend(weights_from_node)
    return weights_list


# weight / 128 -> should be / 2, but we are dividing by larger number to lose float information
def get_rgb_from_weight(weight: float) -> str:
    weight += 1
    weight = weight / 128
    weight = 255 - int(weight * 255) * 64

    return "rgb({0},{0},{0})".format(weight)


def get_weight_group(min_range: float, max_range: float, weights: list, edge_positions: list, threshold: float) \
        -> Tuple[list, list]:
    edge_pos_x = []
    edge_pos_y = []
    for index, weight in enumerate(weights):
        if min_range < weight < max_range and abs(weight) >= threshold:
            edge_pos_x.extend([*edge_positions[index][0], None])
            edge_pos_y.extend([*edge_positions[index][1], None])
    return edge_pos_x, edge_pos_y


def get_min_weight(weights: list) -> float:
    minimum = 1
    for weight in weights:
        minimum = min(minimum, np.amin(weight))
    return minimum


def get_max_weight(weights: list) -> float:
    maximum = -1
    for weight in weights:
        maximum = max(maximum, np.amax(weight))
    return maximum


def visualise_ML(layers: list, bias: list, weights: list, epoch_number: int, steps_size: int = 1, threshold: float = 0):
    # Create figure
    fig = go.Figure(layout={"width": WINDOW_WIDTH, "height": WINDOW_HEIGHT})

    dist_x, dist_y = calc_distance_between_nodes(layers)

    node_position_list_x, node_position_list_y = create_node_position_list(layers, dist_x, dist_y)

    edge_positions = create_edge_position_list(layers, node_position_list_x, node_position_list_y)
    groups = 11
    for epoch in range(0, epoch_number, steps_size):
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
                colorscale='YlGnBu',
                reversescale=True,
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
            args=[{"visible": [False] * len(fig.data)}],
        )

        for j in range(scatter_number_per_epoch):
            step["args"][0]["visible"][i + j] = True

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

    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    app.run_server(debug=True, use_reloader=False)
