from typing import Dict, List, Any
from math import ceil
from npuperf.classes.workload.json_parser.parser import Parser


class MatmulParser(Parser):
    """Matmul 的 parser
    从 HHB networks 中的 'qnn.csi.matmul' 算子类型来看, matmul 有两种情况:
    1. 张量和权重乘, 这种情况下张量的Batch 和权重的Batch可能不相等, 而权重的Batch 一般都是 1 (也就是说, 权重本质上是二维矩阵), 输出张量的Batch == 输入张量的Batch
    2. 张量和张量乘, 这种情况下两个输入张量和输出张量的Batch 都是相等的
    """

    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

        # 取 Mapping
        if mapping.get('Matmul'):
            self.mapping = mapping['Matmul']
        else:
            raise ValueError(f'Matmul Mapping NOT Found! Check mapping dict or path.')
        
        assert len(self.inputs_li) == 3, f'Matmul layer, the number of inputs is not 3: len={len(self.inputs_li)}, names={[ins["name"] for ins in self.inputs_li]}'
        self.input_dim_C = self.extract_dim_C()

    def run(self):
        d = {}
        d['operator_type'] = 'Matmul'
        d['equation_relations'] = []
        d['loop_dim_size'] = {
            'B': self.output_dim_B,
            'K': self.output_dim_K,
            'C': self.input_dim_C,
            'OX': self.output_dim_OX,
        }
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d['core_allocation'] = self.mapping['core_allocation']
        d['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d['loop_dim_size'])

        if self.check_type_of_matmul_is_tensor_x_weight():
            d['equation'] = "O[b][k][ox]+=W[k][c]*I[b][c][ox]"
            d['operand_precision'] = {'O': self.partial_sum_precision, 'O_final': self.data_precision, 'W': self.data_precision, 'I': self.data_precision}
            d['operand_source'] = {'W': [], 'I': IFM_name}
            d['constant_operands'] = ['W']
            d['memory_operand_links'] = self.mapping['memory_operand_links']['OWI']

        else:
            d['equation'] = "O[b][k][ox]+=X[b][k][c]*Y[b][c][ox]"
            d['operand_precision'] = {'O': self.partial_sum_precision, 'O_final': self.data_precision, 'X': self.data_precision, 'Y': self.data_precision}
            d['operand_source'] = {'X': [IFM_name[0]], 'Y': [IFM_name[1]]}
            d['constant_operands'] = []
            d['memory_operand_links'] = self.mapping['memory_operand_links']['OXY'] # todo better check

        self.d = d
        return self.d

    def extract_dim_C(self):
        """        
        取输入特征图的C维度, C 即矩阵相乘后消失的中间维度。要从 inputs 中抓取，并且判断出 C 维度
        C 是从第一个输入的tensor中提取的
        """
        IN_1 = self.inputs_li[0]
        assert IN_1["layout"] == "NCW", f'Matmul layer, inputs format is not NCW, it is {IN_1["layout"]} format!, input name={IN_1["name"]}'
        return IN_1["dim"][2]

    def check_type_of_matmul_is_tensor_x_weight(self):
        """检查matmul 的类型
        矩阵 x 矩阵 返回 False
        矩阵 x 权重 返回 True
        """
        if self.inputs_li[1]["is_const"]:
            # 如果是张量x权重，那么检查权重的batch=1（是二维的）
            assert self.inputs_li[1]["dim"][0] == 1, f'constant input of matmul is not 2-dimension!, dim={self.inputs_li[1]["dim"]}'
            return True
        else:
            return False
