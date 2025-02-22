from typing import Dict, List, Any
from math import ceil
from npuperf.classes.workload.json_parser.parser import Parser


class ConvParser(Parser):
    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

        # 取 Mapping
        if mapping.get('Conv'):
            self.mapping = mapping['Conv']
        else:
            raise ValueError(f'Conv Mapping NOT Found! Check mapping dict or path.')

        group = self.attrs_di['groups']
        self.stride_height, self.stride_width = self.attrs_di['strides']
        self.dilation_height, self.dilation_width = self.attrs_di['dilation']

        # 取 kernel size
        kernel = self.inputs_li[1]
        if ('OIHW' == kernel['layout']) or ('O1HW' == kernel['layout']):
            kernel_dim_C, kernel_dim_FY, kernel_dim_FX = kernel['dim'][1:]
            self.kernel_dim_C = ceil(kernel_dim_C / group)
            self.kernel_dim_FX = kernel_dim_FX
            self.kernel_dim_FY = kernel_dim_FY
        else:
            raise ValueError(f'Conv layer, kernel dim of {kernel["dim"]} is Not OIHW or O1HW format! It is {kernel["layout"]} format!')

    def run(self):
        d = {}
        d['operator_type'] = 'Conv'
        assert (self.kernel_dim_FY == self.kernel_dim_FX), "Conv layer, kernel size ERROR, FX != FY"
        if self.kernel_dim_FY > 1:
            equation = 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]'
        else:   # kernel size 1x1
            equation = 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]'
        d['equation'] = equation
        d['equation_relations'] = [f'ix={self.stride_width}*ox+{self.dilation_width}*fx', f'iy={self.stride_height}*oy+{self.dilation_height}*fy']
        d['loop_dim_size'] = {'B': self.output_dim_B, 'K': self.output_dim_K, 'C': self.kernel_dim_C, 'OY': self.output_dim_OY, 'OX': self.output_dim_OX, 'FY': self.kernel_dim_FY, 'FX': self.kernel_dim_FX}
        d['operand_precision'] = {'O': self.partial_sum_precision, 'O_final': self.data_precision, 'W': self.data_precision, 'I': self.data_precision}
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d['operand_source'] = {'W': [], 'I': IFM_name}
        d['constant_operands'] = ['W']
        d['core_allocation'] = self.mapping['core_allocation']
        d['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d['loop_dim_size'])
        d['memory_operand_links'] = self.mapping['memory_operand_links']

        self.d = d
        return self.d


class DeConvParser(Parser):
    """
    deconv 采用拆分kernel 的方法将一个deconv 层转化为若干个卷积层和一个contract 层  \\
    deconv 利用拆分kernel 的方法转换为conv计算:     \\
    deconv 的kernel可以拆分为 (x_stride * y_stride) 份, 然后分别与IFM 进行conv  \\
    得到的每一个OFM 的shape 是完全相同的, 其宽高方向和 IFM 同样大小 \\
    然后把这些OFM 进行 contract, 宽高方向扩大为 最终OFM 本来的size  \\
    OFM宽高和IFM宽高的关系: OFM_H = IFM_H * y_stride, IFM_W = IFM_H * x_stride, 即宽高扩大了stride 倍
    """
    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict], deconv_count):
        super().__init__(layer, mapping)

        self.deconv_count = deconv_count

        # 取 Mapping, 因为要把 deconv 转换为conv，所以取conv 的mapping
        if mapping.get('Conv'):
            self.mapping = mapping['Conv']
        else:
            raise ValueError(f'Conv Mapping NOT Found! Check mapping dict or path.')

        group = self.attrs_di['groups']
        self.stride_height, self.stride_width = self.attrs_di['strides']
        self.dilation_height, self.dilation_width = self.attrs_di['dilation']

        # 对于deconv 的stride，先支持都是 2
        assert (self.stride_height == self.stride_width) and (self.stride_height == 2), f"deconv layer, only support x_stride == y_stride == 2, \
        current stide are: x_stride = {self.stride_width}, y_stride = {self.stride_height}"

        # 取 kernel size
        kernel = self.inputs_li[1]
        if 'IOHW' in kernel['layout']:
            kernel_dim_C, _, kernel_dim_FY, kernel_dim_FX = kernel['dim'][:]
            self.kernel_dim_C = ceil(kernel_dim_C / group)
            self.kernel_dim_FX = kernel_dim_FX
            self.kernel_dim_FY = kernel_dim_FY
            assert self.kernel_dim_FX == self.kernel_dim_FY, " kernel size uncorrect, FX != FY "    # 判断下 kernel 是个正方形，否则不好做
        else:
            raise ValueError(f'De_Conv layer, kernel dim of {kernel["dim"]} is Not IOHW format! It is {kernel["layout"]} format!')

    def run(self):
        """ 不同于其他算子返回的是一层的dict 描述, deconv 在解析时会转化为若干卷积层和一个contract 层, 所以返回一个大的dict, 里面包含所有的conv 和 contract 的dict 描述 """
        d = {}
        # 首先拆kernel
        self.split_kernel()
        # 然后定义每一个小的conv layer
        for i, kernel_size in enumerate(self.kernel_list):
            conv_dict = self.define_conv(kernel_size)
            d[f'conv_{i}'] = conv_dict
        # 然后定义 contract layer
        d['contract'] = self.define_contract()
        # 最后返回大的dict
        self.d = d
        return self.d

    def split_kernel(self):
        """
        现在只考虑拆成4 份 (stride = 2)
        e.g.1 : 9x9 kernel and stride = 2 --> [(5, 5), (5, 4), (4, 5), (4, 4)]    \\
        e.g.2 : 3x3 kernel and stride = 2 --> [(2, 2), (2, 1), (1, 2), (1, 1)]
        """
        big = ceil(self.kernel_dim_FY / 2)
        small = big - 1
        self.kernel_list = [(big, big), (big, small), (small, big), (small, small)]

    def define_conv(self, kernel_size: tuple):
        fy, fx = kernel_size
        d_conv = {}
        d_conv['operator_type'] = 'Conv_from_deconv'
        if (fy > 1) and (fx > 1):
            equation = 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]'
        elif (fy == 1) and (fx == 1):   # kernel size 1x1
            equation = 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]'
        elif (fy > 1) and (fx == 1):
            equation = 'O[b][k][oy][ox]+=W[k][c][fy]*I[b][c][ix][iy]'
        else:   # (fy == 1) and (fx > 1)
            equation = 'O[b][k][oy][ox]+=W[k][c][fx]*I[b][c][ix][iy]'
        d_conv['equation'] = equation
        d_conv['equation_relations'] = [f'ix=1*ox+{self.dilation_width}*fx', f'iy=1*oy+{self.dilation_height}*fy']
        d_conv['loop_dim_size'] = {'B': self.output_dim_B, 'K': self.output_dim_K, 'C': self.kernel_dim_C, 'OY': int(self.output_dim_OY / self.stride_height), 'OX': int(self.output_dim_OX / self.stride_width), 'FY': fy, 'FX': fx}
        d_conv['operand_precision'] = {'O': self.partial_sum_precision, 'O_final': self.data_precision, 'W': self.data_precision, 'I': self.data_precision}
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d_conv['operand_source'] = {'W': [], 'I': IFM_name}
        d_conv['constant_operands'] = ['W']
        d_conv['core_allocation'] = self.mapping['core_allocation']
        d_conv['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d_conv['loop_dim_size'])
        d_conv['memory_operand_links'] = self.mapping['memory_operand_links']

        return d_conv

    def define_contract(self):
        d_contract = {}
        d_contract['operator_type'] = 'Contract'
        d_contract['equation'] = 'contract'
        d_contract['loop_dim_size'] = {'B': self.output_dim_B, 'G': self.output_dim_K, 'OY': self.output_dim_OY, 'OX': self.output_dim_OX}
        d_contract['operand_precision'] = {'O': self.data_precision, 'O_final': self.data_precision, 'I': self.data_precision}
        IFM_name = [f'contract_{self.deconv_count}_{index}' for index in range(1, self.stride_width * self.stride_height + 1)]
        d_contract['operand_source'] = self.gen_op_source(IFM_name, self.stride_height * self.stride_width)
        d_contract['constant_operands'] = []
        d_contract['core_allocation'] = 1          # 这里先设为固定值 1，只考虑单核的情况
        d_contract['memory_operand_links'] = {'O': 'O', 'I': 'I1'}       # 这里 concat层不使用mapping，直接添加

        return d_contract

    def gen_op_source(self, IFM_name_list, inputs_count):
        op_source = {}
        for id in range(inputs_count):
            op_source[f'X{id}'] = [IFM_name_list[id]]
        return op_source


class dwConvParser(Parser):
    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

        # 取 Mapping
        if mapping.get('dw_Conv'):
            self.mapping = mapping['dw_Conv']
        else:
            raise ValueError(f'dw_Conv Mapping NOT Found! Check mapping dict or path.')

        self.stride_height, self.stride_width = self.attrs_di['strides']
        self.dilation_height, self.dilation_width = self.attrs_di['dilation']

        # 取 kernel size
        kernel = self.inputs_li[1]
        if 'O1HW' in kernel['layout']:
            _, _, kernel_dim_FY, kernel_dim_FX = kernel['dim'][:]
            self.kernel_dim_FX = kernel_dim_FX
            self.kernel_dim_FY = kernel_dim_FY
        else:
            raise ValueError(f'dw Conv layer, kernel dim of {kernel["dim"]} is Not O1HW format! It is {kernel["layout"]} format!')

    def run(self):
        d = {}
        d['operator_type'] = 'dw_Conv'
        assert (self.kernel_dim_FY == self.kernel_dim_FX), "dw Conv layer, kernel size ERROR, FX != FY"
        if self.kernel_dim_FY > 1:
            equation = 'O[b][g][oy][ox]+=W[g][fy][fx]*I[b][g][ix][iy]'
        else:   # kernel size 1x1
            equation = 'O[b][g][oy][ox]+=W[g]*I[b][g][ix][iy]'
        d['equation'] = equation
        d['equation_relations'] = [f'ix={self.stride_width}*ox+{self.dilation_width}*fx', f'iy={self.stride_height}*oy+{self.dilation_height}*fy']
        d['loop_dim_size'] = {'B': self.output_dim_B, 'G': self.output_dim_K, 'OY': self.output_dim_OY, 'OX': self.output_dim_OX, 'FY': self.kernel_dim_FY, 'FX': self.kernel_dim_FX}
        d['operand_precision'] = {'O': self.partial_sum_precision, 'O_final': self.data_precision, 'W': self.data_precision, 'I': self.data_precision}
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d['operand_source'] = {'W': [], 'I': IFM_name}
        d['constant_operands'] = ['W']
        d['core_allocation'] = self.mapping['core_allocation']
        d['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d['loop_dim_size'])
        d['memory_operand_links'] = self.mapping['memory_operand_links']

        self.d = d
        return self.d


class DenseParser(Parser):
    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Dict]):
        super().__init__(layer, mapping)

        # 取 Mapping
        if mapping.get('Dense'):
            self.mapping = mapping['Dense']
        else:
            raise ValueError(f'Dense Mapping NOT Found! Check mapping dict or path.')

        # 取 kernel size
        kernel = self.inputs_li[1]
        if 'OI' == kernel['layout']:
            self.kernel_dim_C = kernel['dim'][1]
        else:
            raise ValueError(f'Dense layer, Kernel dim of {kernel["dim"]} is Not OI format! It is {kernel["layout"]} format!')

    def run(self):
        d = {}
        d['operator_type'] = 'Dense'
        d['equation'] = 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]'
        d['equation_relations'] = ['ix=1*ox+1*fx', 'iy=1*oy+1*fy']
        d['loop_dim_size'] = {'B': self.output_dim_B, 'K': self.output_dim_K, 'C': self.kernel_dim_C, 'OY': self.output_dim_OY, 'OX': self.output_dim_OX, 'FY': 1, 'FX': 1}
        d['operand_precision'] = {'O': self.partial_sum_precision, 'O_final': self.data_precision, 'W': self.data_precision, 'I': self.data_precision}
        IFM_name = [ifm['name'] for ifm in self.inputs_li if ifm['is_const'] == 0]
        d['operand_source'] = {'W': [], 'I': IFM_name}
        d['constant_operands'] = ['W']
        d['core_allocation'] = self.mapping['core_allocation']
        d['spatial_mapping'] = Parser.get_spatial_mapping(self.mapping, d['loop_dim_size'])
        d['memory_operand_links'] = self.mapping['memory_operand_links']

        self.d = d
        return self.d

