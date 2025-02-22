from typing import Dict, List, Any
from npuperf.classes.workload.json_parser.parser import Parser


class PoolParser(Parser):
    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

        # 取 Mapping
        if mapping.get('Pool'):
            self.mapping = mapping['Pool']
        else:
            raise ValueError(f'Pool Mapping NOT Found!Check mapping dict or path.')

        # 取 kernel size, 这里要区分 POOL和 GLOBAL POOL
        if 'global' in self.layer['op_type']:
            if 'NCHW' not in self.inputs_li[0]['layout']:
                raise ValueError(f'the Global Pool layer input layout is not NCHW!, check layer: {self.layer["op_type"]}')
            self.kernel_dim_FY = self.inputs_li[0]['dim'][2]
            self.kernel_dim_FX = self.inputs_li[0]['dim'][3]
            self.stride_height = 1
            self.stride_width = 1
        else:
            self.kernel_dim_FY, self.kernel_dim_FX = self.attrs_di['pool_size']
            self.stride_height, self.stride_width = self.attrs_di['strides']

    def run(self):
        d = {}
        d['operator_type'] = 'Pool'
        d['equation'] = 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]'
        d['equation_relations'] = [f'ix={self.stride_width}*ox+1*fx', f'iy={self.stride_height}*oy+1*fy']
        d['loop_dim_size'] = {'B': self.output_dim_B, 'G': self.output_dim_K, 'OY': self.output_dim_OY, 'OX': self.output_dim_OX, 'FY': self.kernel_dim_FY, 'FX': self.kernel_dim_FX}
        d['operand_precision'] = {'O': self.partial_sum_precision, 'O_final': self.data_precision, 'W': 0, 'I': self.data_precision}
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d['operand_source'] = {'W': [], 'I': IFM_name}
        d['constant_operands'] = ['W']
        d['core_allocation'] = self.mapping['core_allocation']
        d['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d['loop_dim_size'])
        d['memory_operand_links'] = self.mapping['memory_operand_links']

        self.d = d
        return self.d

