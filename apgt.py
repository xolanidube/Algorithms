import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import heapq

def predict_future_weights(edge_history, alpha, time_horizon):
    """
    Predict future edge weights using Exponential Smoothing.
    """
    model = ExponentialSmoothing(edge_history, trend=None, seasonal=None)
    model_fit = model.fit(smoothing_level=alpha, optimized=False)
    prediction = model_fit.forecast(time_horizon)
    # Return the average predicted weight over the time horizon
    return np.mean(prediction)

def expected_weight(graph, u, v):
    """
    Get the expected weight of edge (u, v).
    """
    return graph[u][v]['predicted_weight']

def apgt(graph, start, goal, time_horizon=5, alpha=0.5):
    """
    Adaptive Probabilistic Graph Traversal algorithm.
    """
    # Step 1: Predict future weights for all edges
    for u, v in graph.edges():
        edge_history = graph[u][v]['history']
        predicted_weight = predict_future_weights(edge_history, alpha, time_horizon)
        graph[u][v]['predicted_weight'] = predicted_weight

    # Step 2: Initialize data structures
    queue = []
    heapq.heappush(queue, (0, start))
    costs = {node: float('inf') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    costs[start] = 0

    # Step 3: Traversal
    while queue:
        current_cost, current_node = heapq.heappop(queue)
        if current_node == goal:
            break
        for neighbor in graph.neighbors(current_node):
            weight = expected_weight(graph, current_node, neighbor)
            total_cost = current_cost + weight
            if total_cost < costs[neighbor]:
                costs[neighbor] = total_cost
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (total_cost, neighbor))

    # Step 4: Reconstruct path
    path = reconstruct_path(predecessors, start, goal)
    return path, costs[goal]

def reconstruct_path(predecessors, start, goal):
    """
    Reconstruct the path from start to goal.
    """
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = predecessors[node]
    path.reverse()
    if path[0] == start:
        return path
    else:
        return []

# Sample usage with a dynamic graph
def create_sample_graph():
    G = nx.DiGraph()

    # Add nodes
    for i in range(1, 7):
        G.add_node(i)

    # Add edges with historical weights
    G.add_edge(1, 2, history=[5, 4, 6, 5])
    G.add_edge(1, 3, history=[2, 2, 3, 2])
    G.add_edge(2, 4, history=[7, 8, 6, 7])
    G.add_edge(3, 4, history=[4, 3, 5, 4])
    G.add_edge(4, 5, history=[3, 3, 4, 2])
    G.add_edge(5, 6, history=[1, 1, 2, 1])
    G.add_edge(2, 5, history=[10, 9, 11, 10])
    G.add_edge(3, 5, history=[5, 5, 6, 5])

    return G

def visualize_graph(graph, path):
    pos = nx.spring_layout(graph)
    edge_labels = {(u, v): f"{graph[u][v]['predicted_weight']:.2f}" for u, v in graph.edges()}
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.show()

def main():
    G = create_sample_graph()
    start_node = 1
    goal_node = 6
    time_horizon = 5
    alpha = 0.5  # Smoothing factor for Exponential Smoothing

    path, total_cost = apgt(G, start_node, goal_node, time_horizon, alpha)
    print(f"Optimal path from {start_node} to {goal_node}: {path}")
    print(f"Total expected cost: {total_cost:.2f}")

    visualize_graph(G, path)

if __name__ == "__main__":
    main()
