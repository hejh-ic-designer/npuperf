from typing import Dict, List, Any
from npuperf.classes.workload.json_parser.parser import Parser


class ConcatParser(Parser):

    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

    def run(self):
        d = {}
        d['operator_type'] = 'Concat'
        d['equation'] = 'concat'
        d['loop_dim_size'] = {'B': self.output_dim_B, 'G': self.output_dim_K, 'OY': self.output_dim_OY, 'OX': self.output_dim_OX}
        d['operand_precision'] = {'O': self.data_precision, 'O_final': self.data_precision, 'I': self.data_precision}
        IFM_names = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d['operand_source'] = self.gen_op_source(IFM_names)
        d['constant_operands'] = []
        d['core_allocation'] = 1  # 这里先设为固定值 1，只考虑单核的情况
        d['memory_operand_links'] = {'O': 'O', 'I': 'I1'}  # 这里 concat层不使用mapping，直接添加

        self.d = d
        return self.d

    def gen_op_source(self, IFM_names: list):
        op_source = {}
        for id, IFM_name in enumerate(IFM_names):
            op_source[f'X{id}'] = [IFM_name]
        return op_source


class TransposeParser(Parser):

    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

    def run(self):
        d = {}
        d['operator_type'] = 'Tranpose'
        d['equation'] = 'transpose'
        d['loop_dim_size'] = {
            'B': self.output_dim_B,
            'L': self.output_dim_L,
            'G': self.output_dim_K,
            'D': self.output_dim_D,
            'OY': self.output_dim_OY,
            'OX': self.output_dim_OX
        }
        d['operand_precision'] = {'O': self.data_precision, 'O_final': self.data_precision, 'I': self.data_precision}
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d['operand_source'] = {'I': IFM_name}
        d['constant_operands'] = []
        d['core_allocation'] = 1  # 这里先设为固定值 1，只考虑单核的情况
        d['memory_operand_links'] = {'O': 'O', 'I': 'I1'}  # 这里 concat层不使用mapping，直接添加

        self.d = d
        return self.d


class SplitParser(Parser):

    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)
        self.input_dim_B, self.input_dim_C, self.input_dim_IY, self.input_dim_IX = self.output_dim_B, self.output_dim_K, self.output_dim_OY, self.output_dim_OX

    def run(self):
        d = {}
        d['operator_type'] = 'Split'
        d['equation'] = 'split'
        d['loop_dim_size'] = {
            'B': self.input_dim_B,
            'G': self.input_dim_C,
            'OY': self.input_dim_IY,
            'OX': self.input_dim_IX
        }  # 这里应该写成 IFM 的各个维度，因为 OFM 有多个 (在Parser<class>中提取的dim_K实际上是IFM的NCW中的C，所以变量名为output，实际上是input)
        d['operand_precision'] = {'O': self.data_precision, 'O_final': self.data_precision, 'I': self.data_precision}
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d['operand_source'] = {'I': IFM_name}
        d['constant_operands'] = []
        d['core_allocation'] = 1  # 这里先设为固定值 1，只考虑单核的情况
        d['memory_operand_links'] = {'O': 'O', 'I': 'I1'}  # 这里 split 层不使用mapping，直接添加

        self.d = d
        return self.d
