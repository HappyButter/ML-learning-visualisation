WINDOW_HEIGHT = 600
WINDOW_WIDTH = 900


def calc_distance_between_nodes(layers):
    amount_of_layers = len(layers) + 2
    dist_x = WINDOW_WIDTH // amount_of_layers
    dist_y = []
    for layer in layers:
        dist_y.append(WINDOW_HEIGHT // (layer + 2))
    return dist_x, dist_y


def create_node_position_list(layers, dist_x, dist_y):
    amount_of_nodes = sum(layers)
    node_position_list_x = []
    distance_counter = 0
    for layer in layers:
        distance_counter += dist_x
        node_position_list_x.extend([distance_counter for _ in range(layer)])

    node_position_list_y = []
    for number_of_layer, layer_size in enumerate(layers):
        node_position_list_y.extend([dist_y[number_of_layer] * (x + 1) for x in range(layer_size)])

    return node_position_list_x, node_position_list_y


def create_edge_position_list(layers, node_position_list_x, node_position_list_y):
    edge_position_list_x = []
    edge_position_list_y = []
    current_pos = layers[0]
    amount_of_layers = len(layers) - 1
    for i in range(amount_of_layers):

        for j in range(layers[i + 1]):
            x1 = node_position_list_x[current_pos + j]
            x2 = 0
            y1 = node_position_list_y[current_pos + j]
            y2 = 0
            for k in range(layers[i]):
                x2 = node_position_list_x[current_pos - layers[i] + k]
                y2 = node_position_list_y[current_pos - layers[i] + k]
                edge_position_list_x.append(x1)
                edge_position_list_x.append(x2)
                edge_position_list_x.append(None)
                edge_position_list_y.append(y1)
                edge_position_list_y.append(y2)
                edge_position_list_y.append(None)
        current_pos += layers[i + 1]

    return edge_position_list_x, edge_position_list_y


def create_bias_list(layers, bias):
    bias_list = [0 for _ in range(layers[0])]
    for layer_bias in bias:
        bias_list.extend(layer_bias)

    return bias_list

def create_weights_list(layers, weights):
    weights_list = []
    for weights_layer in weights:
        for weights_from_node in weights_layer:
            weights_list.extend(weights_from_node)
    return weights_list


# weight / 128 -> should be / 2, but we are dividing by larger number to lose float information
def get_rgb_from_weight(weight):
    weight += 1
    weight = weight / 128
    weight = 255 - int(weight * 255) * 64

    return "rgb({0},{0},{0})".format(weight)