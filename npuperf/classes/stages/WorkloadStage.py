import logging

import networkx as nx

from npuperf.classes.stages.Stage import Stage
from npuperf.classes.workload.layer_node import InputLayerNode, LayerNode
from npuperf.classes.workload.mem_node import MemNode
from npuperf.utils import pickle_deepcopy

logger = logging.getLogger(__name__)


class WorkloadStage(Stage):
    """
    Class that iterates through the nodes in a given workload graph.
    """

    def __init__(self, list_of_callables, *, workload: nx.DiGraph, execution_layers=None, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.execution_layers = execution_layers
        if execution_layers:
            logger.info(f'Running these layers with id: {execution_layers}')

    # 如果是普通的层，那就原样往下跑
    # 如果是不用上PE array的，那就直接从这里跳到最后一个stage，即 costmodel stage，直接进行data copy 评估
    def run(self):
        for layer in nx.topological_sort(self.workload):

            if isinstance(layer, InputLayerNode) or ((self.execution_layers) and (layer.id not in self.execution_layers)):
                continue

            kwargs = self.kwargs.copy()
            kwargs['layer'] = layer

            logger.info(f"Current layer is {layer.TYPE}, at id {layer.id} " + "---" * 30)
            logger.info(f'layer size: {layer.loop_dim_size}')
            if isinstance(layer, LayerNode):
                if (layer.TYPE in ['Add', 'Subtract', 'Mul', 'Matmul'] or layer.TYPE is None) and layer.memory_operand_links.get("X", False):
                    # 这里主要是add 层和mul 层，workload中定义的X和Y 都来自mem 中的I1，但这样的话会在下一层中报错
                    # 所以应该在实际的mem hier中把所有I2 全部remove掉，然后把I1对应的 mem lv加上对应的I2 标签，然后把workload中的 Y 再改成来自I2
                    # 这样的话在下一层才不会报错，而且实际上也表达了 X和Y 都来自实际的I1
                    accelerator_for_add = pickle_deepcopy(kwargs['accelerator'])  # 拷贝出来一份，否则add算子将accelerator处理后，add后面的算子也会受到影响
                    layer, accelerator = self.Add_Mul_operator_process(layer, accelerator_for_add)
                    kwargs['accelerator'] = accelerator
                    kwargs['layer'] = layer

                sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                for cme, extra_info in sub_stage.run():
                    yield cme, (layer, extra_info)

            elif isinstance(layer, MemNode):
                sub_stage = self.list_of_callables[-1]([], **kwargs)
                for cme, extra_info in sub_stage.run():
                    yield cme, (layer, extra_info)

            else:
                raise TypeError("the layer is not type of LayerNode, InputLayerNode, or MemNode.")

    def Add_Mul_operator_process(self, layer: LayerNode, accelerator):
        mh = accelerator.get_core(layer.core_allocation).get_memory_hierarchy()
        while mh.remove_operator_top_level('I2')[0]:  # 把I2 拆光了
            pass

        for ml in mh.nodes:
            if 'I1' in ml.operands:
                ml.operands.append('I2')
                ml.mem_level_of_operands['I2'] = ml.mem_level_of_operands['I1']
                l = list(ml.port_alloc_raw)
                l.append(ml.port_alloc_raw[ml.operands.index('I1')].copy())
                ml.port_alloc_raw = tuple(l)
                for p in (ml.port_list):
                    for sold in p.served_op_lv_dir[:]:
                        if sold[0] == 'I1':
                            p.add_port_function(tuple(['I2'] + list(sold[1:])))
        mh.nb_levels['I2'] = mh.nb_levels['I1']
        accelerator.get_core(layer.core_allocation).recalculate_memory_hierarchy_information()
        layer.memory_operand_links[layer.input_operands[1]] = 'I2'
        return layer, accelerator


class WorkloadStage_ref(Stage):
    """
    Class that iterates through the nodes in a given workload graph.
    """

    def __init__(self, list_of_callables, *, workload, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload

    def run(self):
        for layer in nx.topological_sort(self.workload):
            if isinstance(layer, InputLayerNode):
                continue
            kwargs = self.kwargs.copy()
            kwargs['layer'] = layer
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (layer, extra_info)
