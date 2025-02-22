import networkx as nx
import matplotlib.pyplot as plt
import math

from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy

def visualize_memory_hierarchy_graph(G: MemoryHierarchy):
    """
    Visualizes a memory hierarchy graph.
    """

    generations = list(nx.topological_generations(G))

    # print(generations)

    max_nodes_gen = max((len(generation) for generation in generations))
    pos = {}
    node_list = []
    node_size_list = []
    node_label_dict = {}
    for gen_idx, generation in enumerate(generations):
        y = gen_idx
        node_size = (gen_idx + 1) * 300
        for node_idx, node in enumerate(generation):
            if len(generation) == max_nodes_gen:
                x = node_idx
            else:
                x = (node_idx + 1) * (max_nodes_gen - 1) / (len(generation) + 1)
            pos[node] = (x, y)
            node_list.append(node)
            node_size_list.append(node_size)
            node_label_dict[node] = f"{node.name}\n{node.operands}"

    # nx.draw(G, pos=pos, node_shape='s', nodelist=node_list, node_size=node_size_list, labels=node_label_dict)
    nx.draw_networkx(G, pos=pos, node_shape='s', nodelist=node_list, node_size=node_size_list, labels=node_label_dict)
    plt.title(G.name)
    # plt.tight_layout()
    plt.show()


def visualize_memory_hierarchy_graph_new(G: MemoryHierarchy):
    """
    Visualizes a memory hierarchy graph.
    """

    generations = list(nx.topological_generations(G))

    # print(generations)

    max_nodes_gen = max((len(generation) for generation in generations))
    pos = {}
    node_list = []
    node_size_list = []
    node_label_dict = {}
    for gen_idx, generation in enumerate(generations):
        y = gen_idx
        node_size = (gen_idx + 1) * 300
        for node_idx, node in enumerate(generation):
            # print(node.operands)
            if node.operands == ['I2']:
                x = 1
            elif node.operands == ['I1']:
                x = 2
            elif node.operands == ['I1', 'O']:
                x = 3
            elif node.operands == ['O']:
                x = 4
            else:
                x = 2.5
            pos[node] = (x, y)
            node_list.append(node)
            node_size_list.append(node_size)
            node_label_dict[node] = f"{node.name}\n{node.operands}"

    # nx.draw(G, pos=pos, node_shape='s', nodelist=node_list, node_size=node_size_list, labels=node_label_dict)
    nx.draw_networkx(G, pos=pos, node_shape='s', nodelist=node_list, node_size=node_size_list, labels=node_label_dict)
    plt.title(G.name)
    # plt.tight_layout()
    plt.show()
