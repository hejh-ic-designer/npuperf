from typing import Dict, List, Any
from npuperf.classes.workload.json_parser.parser import Parser


class AddParser(Parser):

    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

        # 取 Mapping
        if mapping.get('Add'):
            self.mapping = mapping['Add']
        else:
            raise ValueError(f'Add Mapping NOT Found! Check mapping dict or path.')

    def run(self):
        d = {}
        d['operator_type'] = 'Add'
        d['equation_relations'] = []
        d['loop_dim_size'] = {
            'B': self.output_dim_B,
            'G': self.output_dim_K,
            'D': self.output_dim_D,
            'OY': self.output_dim_OY,
            'OX': self.output_dim_OX
        }
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        if len(IFM_name) == 2:
            d['equation'] = Parser.get_equation_XYO(self.inputs_li[0], '+', self.inputs_li[1])
            d['operand_source'] = {'X': [IFM_name[0]], 'Y': [IFM_name[1]]}
            d['constant_operands'] = []
            d['operand_precision'] = {
                'O': self.partial_sum_precision,
                'O_final': self.data_precision,
                'X': self.data_precision,
                'Y': self.data_precision
            }
            d['memory_operand_links'] = self.mapping['memory_operand_links']['OXY']

        elif len(IFM_name) == 1:  # 权重的数据量可能和tensor不等，这里的情况较复杂，要解析权重的layout 和dim，然后去配置对应的equation
            d['equation'] = Parser.get_equation_WIO(self.inputs_li[1], '+', self.inputs_li[0])
            d['operand_source'] = {'I': IFM_name, 'W': []}
            d['constant_operands'] = ['W']
            d['operand_precision'] = {
                'O': self.partial_sum_precision,
                'O_final': self.data_precision,
                'W': self.data_precision,
                'I': self.data_precision
            }
            d['memory_operand_links'] = self.mapping['memory_operand_links']['OWI']
        else:
            raise ValueError(f'ADD operator has INCORRECT number of IFMs: {len(IFM_name)}')
        d['core_allocation'] = self.mapping['core_allocation']
        d['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d['loop_dim_size'])

        self.d = d
        return self.d


class SubtractParser(Parser):

    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

        # 取 Mapping
        if mapping.get('Subtract'):
            self.mapping = mapping['Subtract']
        else:
            raise ValueError(f'Subtract Mapping NOT Found! Check mapping dict or path.')

    def run(self):
        d = {}
        d['operator_type'] = 'Subtract'
        d['equation_relations'] = []
        d['loop_dim_size'] = {
            'B': self.output_dim_B,
            'G': self.output_dim_K,
            'D': self.output_dim_D,
            'OY': self.output_dim_OY,
            'OX': self.output_dim_OX
        }
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]

        ### 目前遇到的 subtract 都是两特征输入，不清楚是否存在 权重输入 的情况，所以先添加这一行 assert 代码
        # by gyp
        #assert len(IFM_name) == 2, f'The IFMs of Subtract layer is not 2!, please check {self.layer["name"]}'
        ### 如果出现 权重输入 的情况，删去这行即可

        if len(IFM_name) == 2:
            d['equation'] = Parser.get_equation_XYO(self.inputs_li[0], '-', self.inputs_li[1])
            d['operand_source'] = {'X': [IFM_name[0]], 'Y': [IFM_name[1]]}
            d['constant_operands'] = []
            d['operand_precision'] = {
                'O': self.partial_sum_precision,
                'O_final': self.data_precision,
                'X': self.data_precision,
                'Y': self.data_precision
            }
            d['memory_operand_links'] = self.mapping['memory_operand_links']['OXY']

        elif len(IFM_name) == 1:  # 权重的数据量可能和tensor不等，这里的情况较复杂，要解析权重的layout 和dim，然后去配置对应的equation
            # by gyp 这里会报错，因为get_equation_WIO 第一个输入需要常量weights，但这里input_list索引为0的是weight常量
            # d['equation'] = Parser.get_equation_WIO(self.inputs_li[1], '-', self.inputs_li[0])
            d['equation'] = Parser.get_equation_WIO(self.inputs_li[0], '-', self.inputs_li[1])
            d['operand_source'] = {'I': IFM_name, 'W': []}
            d['constant_operands'] = ['W']
            d['operand_precision'] = {
                'O': self.partial_sum_precision,
                'O_final': self.data_precision,
                'W': self.data_precision,
                'I': self.data_precision
            }
            d['memory_operand_links'] = self.mapping['memory_operand_links']['OWI']
        else:
            raise ValueError(f'ADD operator has INCORRECT number of IFMs: {len(IFM_name)}')
        d['core_allocation'] = self.mapping['core_allocation']
        d['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d['loop_dim_size'])

        self.d = d
        return self.d


class MulParser(Parser):

    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

        # 取 Mapping
        if mapping.get('Mul'):
            self.mapping = mapping['Mul']
        else:
            raise ValueError(f'Mul Mapping NOT Found! Check mapping dict or path.')
        pass

    def run(self):
        d = {}
        d['operator_type'] = 'Mul'
        d['equation_relations'] = []
        d['loop_dim_size'] = {
            'B': self.output_dim_B,
            'G': self.output_dim_K,
            'D': self.output_dim_D,
            'OY': self.output_dim_OY,
            'OX': self.output_dim_OX
        }
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        if len(IFM_name) == 2:
            d['equation'] = Parser.get_equation_XYO(self.inputs_li[0], '*', self.inputs_li[1])
            d['operand_source'] = {'X': [IFM_name[0]], 'Y': [IFM_name[1]]}
            d['constant_operands'] = []
            d['operand_precision'] = {
                'O': self.partial_sum_precision,
                'O_final': self.data_precision,
                'X': self.data_precision,
                'Y': self.data_precision
            }
            d['memory_operand_links'] = self.mapping['memory_operand_links']['OXY']

        elif len(IFM_name) == 1:
            d['equation'] = Parser.get_equation_WIO(self.inputs_li[1], '*', self.inputs_li[0])
            d['operand_source'] = {'I': IFM_name, 'W': []}
            d['constant_operands'] = ['W']
            d['operand_precision'] = {
                'O': self.partial_sum_precision,
                'O_final': self.data_precision,
                'W': self.data_precision,
                'I': self.data_precision
            }
            d['memory_operand_links'] = self.mapping['memory_operand_links']['OWI']
        else:
            raise ValueError(f'MUL operator has INCORRECT number of IFMs: {len(IFM_name)}')
        d['core_allocation'] = self.mapping['core_allocation']
        d['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d['loop_dim_size'])

        self.d = d
        return self.d
