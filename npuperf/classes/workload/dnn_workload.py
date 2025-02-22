import networkx as nx

from npuperf.classes.workload.layer_node import LayerNode, InputLayerNode
from npuperf.classes.workload.mem_node import MemNode
from typing import Dict, Any
from networkx import DiGraph


class DNNWorkload(DiGraph):

    def __init__(self, workload: Dict[Any, Dict], **attr):
        """
        Collect all the algorithmic workload information here.
        :param workload: user-defined workload file (.py).
        :return (self): Directed Graph with nodes the layers and edges the connections between layers.
        """
        super().__init__(**attr)

        layer_id_to_obj = {}  # Lookup dict for id to LayerNode object translation
        #self.layer_node_list = []

        for layer_id, layer in workload.items():
            # TODO Support other type of layers, such as concatenation, max pooling, BN, etc.
            #  What is special about max pooling?
            # elif type(layer_id) == str and layer_id[0:6] == 'concat':
            #     continue
            '''For each item in the dict generate the LayerNode and add it to the dnn graph G'''
            if layer['equation'] == 'input':
                layer_node = InputLayerNode(layer_id, **layer)
                self.add_node(layer_node)
                layer_id_to_obj[layer_id] = layer_node
                #self.layer_node_list.append(layer_node)
            else:

                if layer['equation'] in ['transpose', 'concat', 'contract', 'split']:
                    layer_node = MemNode(layer_id, layer)
                    layer_id_to_obj[layer_id] = layer_node
                    # self.add_node(layer_id, info=layer_node)
                    self.add_node(layer_node)
                    #self.layer_node_list.append(layer_node)
                    '''Find all of its operand sources and add edges accordingly'''
                    edges = []
                    operand_source: Dict[str, list] = layer.get('operand_source', {})
                    operand_source_dimension_mapping: Dict[str, Dict[str, str]] = layer.get('operand_source_dimension_mapping', {})
                    for (op, parent_list) in operand_source.items():
                        parent_id = parent_list[0]
                        parent_layer = layer_id_to_obj[parent_id]
                        edges.append((parent_layer, layer_node))
                        layer_node.input_operand_source[op] = parent_layer      # op: X1,X2,X3 ... or I     # parent_layer: class Layernode or MemNode

                    # check validation 利用 input_operand_source和 operand_source_dimension_mapping检查当前层定义是否合规
                    if layer['equation'] == 'concat':
                        assert self.check_validation_concat(layer_node.input_operand_source, operand_source_dimension_mapping), 'Concat Layer validation FALSE!!!, please check concat layer.'
                    elif layer['equation'] == 'transpose':
                        assert self.check_validation_transpose(layer_node.input_operand_source, operand_source_dimension_mapping), 'Transpose Layer validation FALSE!!!, please check transpose layer.'
                    # 把当前节点的边，和parent layers 连接起来
                    self.add_edges_from(edges)


                else:   # Conv, dw Conv, Pool, Add, pixel shuffle conv
                    layer_node = LayerNode(layer_id, layer)
                    '''Save this layer_id and LayerNode pair in the layer_id_to_obj dict'''

                    layer_id_to_obj[layer_id] = layer_node
                    # self.add_node(layer_id, info=layer_node)
                    self.add_node(layer_node)
                    #self.layer_node_list.append(layer_node)
                    '''Find all of its operand sources and add edges accordingly'''

                    edges = []
                    for (op, parent_list) in layer.get('operand_source', {}).items():
                        for parent_id in parent_list:
                            parent_layer = layer_id_to_obj[parent_id]
                            edges.append((parent_layer, layer_node))
                            layer_node.input_operand_source[op] = parent_layer

                    # 把当前节点的边，和parent layers 连接起来
                    self.add_edges_from(edges)

    def check_validation_concat(self, input_operand_source: Dict[str, Any], mapping: Dict[str, Dict[str, str]]):
        # 入参如：
        # input_operand_source: {'X1': LayerNode_0, 'X2': MemNode_transpose_1, 'X3': LayerNode_2}
        # mapping: {'X1': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}, 'X2': {'OX': 'OX', 'OY': 'OY', 'G': 'G'}, 'X3': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}}
        # TODO

        return True

    def check_validation_transpose(self, input_operand_source: Dict[str, Any], mapping: Dict[str, Dict[str, str]]):
        # 入参如：
        # 'input_operand_source': {'I': LayerNode_0}
        # mapping: {'I': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}}
        # TODO

        return True

    def topological_sort(self):
        return nx.topological_sort(self)

    def get_node_with_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node
        raise ValueError("DNNWorkload instance does not have a node with the requested id")

if __name__ == "__main__":
    from npuperf.visualization.graph.dnn import visualize_dnn_graph
    from npuperf.inputs.WL_fromjson.Meta_prototype.workload_mv1 import workload

    ml_workload = DNNWorkload(workload)

    # print layer size
    weight_size = 0
    activation_size = 0
    I_size_list = []
    O_size_list = []
    MAC_count = 0
    for idx, layer in enumerate(ml_workload.nodes):
        print ()
        if isinstance(layer, InputLayerNode | MemNode):
            continue
        try:
            weight_size += layer.operand_size_elem['W'] * layer.operand_precision['W'] / 8 / 1024
            activation_size += layer.operand_size_elem['I'] * layer.operand_precision['I'] / 8 / 1024
            I_size_list.append(layer.operand_size_elem['I'] * layer.operand_precision['I'] / 8 / 1024)
            O_size_list.append(layer.operand_size_elem['O'] * layer.operand_precision['O_final'] / 8 / 1024)
            MAC_count += layer.total_MAC_count
        except:
            weight_size += 0
        for operand in layer.operand_list:
            if operand == 'O':
                operand1 = 'O_final'
            else:
                operand1 = operand
            print(layer,
                  ' operand', operand,
                  ' size (kB): ', layer.operand_size_elem[operand] * layer.operand_precision[operand1] / 8 / 1024)

    activation_size += layer.operand_size_elem['O'] * layer.operand_precision['O_final'] / 8 / 1024
    print('\nTotal weight size (kB): ', weight_size)
    print('\nTotal activation size (kB): ', activation_size)
    print('\nInput size (kB): ', I_size_list)
    print('Output size (kB): ', O_size_list)
    print('\nAverage Input size (kB): ', sum(I_size_list)/len(I_size_list))
    print('Average Output size (kB): ', sum(O_size_list)/len(O_size_list))
    print('\nMax Input size (kB): ', max(I_size_list))
    print('Max Output size (kB): ', max(O_size_list))

    print('\nTotal MAC count: ', MAC_count)
    G = ml_workload
    visualize_dnn_graph(G)


