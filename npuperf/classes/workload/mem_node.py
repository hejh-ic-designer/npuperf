from typing import Dict, List
from math import prod

class MemNode:
    def __init__(self, layer_id, layer_attrs: dict):
        self.id = layer_id

        '''Get required attributes from layer_attrs'''
        TYPE: str = layer_attrs.get('operator_type', None)
        name:str = layer_attrs.get('name', None)
        equation: str = layer_attrs.get('equation')
        loop_dim_size: Dict[str, int] = layer_attrs.get('loop_dim_size')
        operand_precision: Dict[str, int] = layer_attrs.get('operand_precision')
        equation_relations: List[str] = layer_attrs.get('equation_relations', [])
        core_allocation: int = layer_attrs.get('core_allocation', None)
        memory_operand_links: Dict[str, str] = layer_attrs.get('memory_operand_links', None)
        source_storage_level: int = layer_attrs.get('source_storage_level', {})
        operand_source_dimension_mapping: Dict[Dict[str, str]] = layer_attrs.get('operand_source_dimension_mapping', {})
        constant_operands: List[str] = layer_attrs.get('constant_operands', [])
        post_process: str = layer_attrs.get('post_process', None)

        ''' self attributes '''
        self.TYPE = TYPE
        self.name = name
        self.equation = equation
        self.loop_dim_size = dict(item for item in tuple(loop_dim_size.items()))  # 其实啥都没做，loop_dim_size没变
        self.operand_precision = operand_precision
        self.equation_relations = equation_relations
        self.original_equation_relations = equation_relations.copy()    # equation_relations 的备份
        self.loop_dim_list = list(loop_dim_size.keys())     # 循环变量列表
        self.core_allocation = core_allocation
        self.memory_operand_links = memory_operand_links.copy()
        self.source_storage_level = source_storage_level
        self.operand_source_dimension_mapping = operand_source_dimension_mapping
        self.constant_operands = constant_operands
        self.input_operand_source = dict()      
        self.post_process = post_process  
        self.extract_OFM_info()

    def __str__(self):
        if self.TYPE:
            return f"MemNode_{self.id}_{self.TYPE}"
        return f"MemNode_{self.id}"

    def __repr__(self):
        return str(self)

    def extract_OFM_info(self):
        self.OFM_size = prod(self.loop_dim_size.values())
        self.operand_size_elem = {'O': self.OFM_size}
        self.operand_size_bit = {'O': self.operand_precision['O_final'] * self.operand_size_elem['O']}


if __name__ == '__main__':
    ld_t = { 'operator_type': 'Transpose',  # 在网络描述中，假设什么都不做，1x32x112x112 -> 1x32x112x112，但是在后层执行中计算data_copy的数据量  
           'equation': 'transpose',
           'equation_relations': [],
           'loop_dim_size': {'B': 1, 'G': 32, 'OX': 112, 'OY': 112},
           'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
           'operand_source': {'I': [0]},
           'constant_operands': [],
           'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}},
           'core_allocation': 1,
           'spatial_mapping': {},
           'memory_operand_links': {'O': 'O', 'I': 'I1'}}
    
    ld_cc = {   
            'operator_type': 'Concat',  # 在网络描述中，输入是若干OX OY相同的特征图，但是通道数不一定相同，分别用X1, X2, ... 表示，输出为 O
           'equation': 'concat',
           'equation_relations': [],
           'loop_dim_size': {'B': 1, 'G': 32, 'OX': 112, 'OY': 112},
           'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
           'operand_source': {'X1': [0], 'X2': [1], 'X3': [2] },    # 第0 1 2 层的concat
           'constant_operands': [],
           'operand_source_dimension_mapping': {'X1': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}, 'X2': {'OX': 'OX', 'OY': 'OY', 'G': 'G'}, 'X3': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}},   # 
           'core_allocation': 1,
           'spatial_mapping': {},
           'memory_operand_links': {'O': 'O', 'I': 'I1'}}


    nd = MemNode(1, ld_cc)
    print(nd.TYPE)
    print(nd.source_storage_level)
    print(nd.constant_operands)
    print(nd.core_allocation)
    print(nd.equation)
    print(nd.equation_relations)
    print(nd.id)
    print(nd.loop_dim_list)     # ['B', 'G', 'OX', 'OY']
    print(nd.loop_dim_size)     # {'B': 1, 'G': 32, 'OX': 112, 'OY': 112}
    print(nd.memory_operand_links)
    print(nd.operand_precision)
    print(nd.operand_size_bit)      # dict  {'O': 3211264}      # 下面的数值乘以 O_final precision，一般是 8
    print(nd.operand_size_elem)     # dict  {'O': 401408}       # 实际上就是把loop_dim_size的各数值全乘起来
    print(nd.operand_source_dimension_mapping)
    print(nd.original_equation_relations)
    print(nd.OFM_size)
    print(nd.__str__())     # MemNode_1
    print(nd.__repr__())    # MemNode_1
