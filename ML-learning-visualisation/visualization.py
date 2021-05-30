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





# layers = [4, 20, 20, 3]
# dist_x, dist_y = calc_distance_between_nodes(layers)
# node_position_list_x, node_position_list_y = create_node_position_list(layers, dist_x, dist_y)
# edge_position_list_x, edge_position_list_y = create_edge_position_list(layers, node_position_list_x, node_position_list_y)
# print(edge_position_list_y)
#
# edge_trace = go.Scatter(
#     x=edge_position_list_x, y=edge_position_list_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines')
#
#
# node_trace = go.Scatter(
#     x=node_position_list_x, y=node_position_list_y,
#     mode='markers',
#     hoverinfo='text',
#     marker=dict(
#         showscale=True,
#         # colorscale options
#         #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
#         #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
#         #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
#         colorscale='YlGnBu',
#         reversescale=True,
#         color=[],
#         size=10,
#         colorbar=dict(
#             thickness=15,
#             title='Node Connections',
#             xanchor='left',
#             titleside='right'
#         ),
#         line_width=2))
#
# fig = go.Figure(data=[edge_trace, node_trace],
#              layout=go.Layout(
#                 title='<br>Neural network structure visualization with Python',
#                 titlefont_size=16,
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 annotations=[ dict(
#                     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
#                     showarrow=False,
#                     xref="paper", yref="paper",
#                     x=0.005, y=-0.002 ) ],
#                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                 )
# fig.show()