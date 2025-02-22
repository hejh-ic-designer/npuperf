from math import gcd, prod
import re
from collections import defaultdict
from typing import Dict, List, Any
from copy import deepcopy
from collections import defaultdict
from math import ceil


class InputLayerNode:

    def __init__(self, layer_id, loop_dim_size, precision, memory_operand_links, core_allocation, **unused_kwargs):
        self.input_operands = []
        self.constant_operands = []
        self.operand_precision = {'O': precision, 'O_final': precision}
        self.id = layer_id
        self.loop_dim_size = loop_dim_size
        self.input_operand_source = {}
        self.constant_operands = []
        self.extract_layer_info()
        self.memory_operand_links = memory_operand_links
        self.core_allocation = core_allocation
        self.operand_loop_dim = {'O': {'r': tuple(loop_dim_size.keys())}}

    def __str__(self):
        return f"InputLayerNode_{self.id}"

    def __repr__(self):
        return str(self)

    def extract_layer_info(self):
        self.operand_size_elem = {'O': prod(self.loop_dim_size.values())}
        self.operand_size_bit = {'O': self.operand_precision['O_final'] * self.operand_size_elem['O']}


class LayerNode:

    def __init__(self, layer_id, layer_attrs: Dict[str, Any]):
        """
        To construct each layer node, algorithm equation/dimension/indirect relation are parsed.
        This parser collects information of operand, loop dimension, and loop relevance.
        Equal-to-1 loop dimensions are eliminated.

        :param layer_id: The identifier (key) of the layer, as defined in the workload
        :param layer_attrs: contains attributes specified below:
        *equation: core computation equation, e.g. 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'Y[i][j] += A[i][k] * B[k][j]', 'Y[i][j] += A[i][k][l] * B[k][j] * C[l][j]', etc.
        *loop_dim_size: size of each computation loop, e.g. {'B': 1, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28,
        'FY': 1, 'FX': 1, 'G': 1}.
        *equation_relations: for the operand dimension that is not directly a loop dimension,
        a set of specific relation equations between them (operand dimension and loop dimension) is required,
        e.g. ['ix=ox+fx-1', 'iy=oy+fy-1'].
        *core_allocation: the accelerator core on which this layer is executed
        *memory_operand_links: the link between layer operands and where they are stored in the memory hierarchy.

        :return (self)
        ------- directly get from inputs -------
        - loop_dim_size: collection of loop dimension size that >1.
        - operand_precision
        - loop_dim_list, e.g. ['B', 'K', 'C', ...], collection of loop dimension whose size >1.
        - operand_list, e.g. ['W', 'I', 'O']

        ------- operand and loop dimension relation -------
        - operand_loop_dim: operand and loop dimension relationship, e.g.
        operand_loop_dim = {'O': {'r': ['B', 'K', 'OY', 'OX'], 'ir': ['C', 'FX', 'FY'], 'pr': {}},
                            'W': {'r': ['K', 'C', 'FY', 'FX'], 'ir': ['B', 'OX', 'OY'], 'pr': {}},
                            'I': {'r': ['B', 'C'], 'ir': ['K'], 'pr': {'IY': ('OY', 'FY'), 'IX': ('OX', 'FX')}}}

        ------- basic layer information extraction -------
        - total_MAC_count
        - operand_size_elem
        - operand_size_bit
        - operand_data_reuse
        """

        self.id = layer_id
        '''Get required attributes from layer_attrs'''
        TYPE: str = layer_attrs.get('operator_type', None)
        name: str = layer_attrs.get('name', None)
        equation: str = layer_attrs.get('equation')
        loop_dim_size: Dict[str, int] = layer_attrs.get('loop_dim_size')
        operand_precision: Dict[str, int] = layer_attrs.get('operand_precision')
        equation_relations: List[str] = layer_attrs.get('equation_relations', [])
        user_spatial_mapping: Dict[str, tuple] = layer_attrs.get('spatial_mapping', None)
        user_temporal_ordering = layer_attrs.get('temporal_ordering', None)
        core_allocation: int = layer_attrs.get('core_allocation', None)
        memory_operand_links: Dict[str, str] = layer_attrs.get('memory_operand_links', None)
        source_storage_level: int = layer_attrs.get('source_storage_level', {})
        operand_source_dimension_mapping: Dict[Dict[str, str]] = layer_attrs.get('operand_source_dimension_mapping', {})
        constant_operands: List[str] = layer_attrs.get('constant_operands', [])
        post_process: str = layer_attrs.get('post_process')
        activation_function: str = layer_attrs.get('activation_function')

        self.TYPE = TYPE
        self.name = name
        self.equation = equation
        self.loop_dim_size = dict(item for item in tuple(loop_dim_size.items()))  # 其实啥都没做，loop_dim_size没变
        self.operand_precision = operand_precision
        self.equation_relations = equation_relations
        self.original_equation_relations = equation_relations.copy()  # equation_relations 的备份
        self.loop_dim_list = list(loop_dim_size.keys())  # 循环变量列表
        self.user_spatial_mapping = user_spatial_mapping
        self.user_temporal_ordering = user_temporal_ordering
        self.core_allocation = core_allocation
        self.memory_operand_links = memory_operand_links.copy()
        self.source_storage_level = source_storage_level
        self.operand_source_dimension_mapping = operand_source_dimension_mapping
        self.constant_operands = constant_operands
        self.post_process = post_process
        self.activation_function = activation_function
        ''' Step1: extract partially-relevant data dimension and its relation to loop dimensions. '''
        pr_loop, pr_loop_list, equation = self.build_pr_funcs()
        # pr_loop:      {'IX': ['OX', 'FX'], 'IY': ['OY', 'FY']}    卷积的情况
        # pr_loop_list: ['IX', 'OX', 'FX', 'IY', 'OY', 'FY']
        ''' Step2: extract relevant and irrelevant loop dimensions. '''
        operand_loop_dim, operand_loop_dim_reform, operand_list = \
            self.extract_r_ir_loop_info(equation, self.loop_dim_size, pr_loop, pr_loop_list)
        self.operand_loop_dim = operand_loop_dim
        self.operand_loop_dim_reform = operand_loop_dim_reform
        self.output_operand = operand_list[0]
        self.input_operands = operand_list[1:]
        self.operand_list = operand_list
        self.input_operand_source = dict()
        ''' Step3: extract layer info, e.g. total operand size, total operand data reuse, total MAC operation, etc. '''
        self.extract_layer_info()

    def build_pr_funcs(self):
        # 1 long dimensions are removed in self.loop_dim_size but required in extract_pr_loop_info
        loop_dim_size = defaultdict(lambda: 1)  #= 创建一个空dict，查找key 不存在时自动生成对应value 为 1
        loop_dim_size.update(self.loop_dim_size)  # 就是loop_dim_size，相当于copy了一个defaultdict版本
        equation = self.equation
        if self.equation_relations:
            pr_loop, pr_loop_list, pr_scaling_factors = self.extract_pr_loop_info(self.equation_relations)
        else:  # equation_relations == []
            pr_loop, pr_loop_list, pr_scaling_factors = {}, [], {}

        self.pr_loop = pr_loop
        self.pr_loop_list = pr_loop_list
        self.pr_scaling_factors = pr_scaling_factors
        # {'IX': {'ox': 2, 'fx': 1}, 'IY': {'oy': 2, 'fy': 1}}  卷积的情况，下面 if __name__=='__main__' 的结果
        return pr_loop, pr_loop_list, equation

    def get_core_allocation(self):
        return self.core_allocation

    def __str__(self):
        if self.TYPE:
            return f"LayerNode_{self.id}_{self.TYPE}"
        return f"LayerNode_{self.id}"

    def __repr__(self):
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return {
            "name": self.name,
            "equation": self.equation,
            "equation_relations": self.equation_relations,
            "loop_dimensions": self.loop_dim_size,
            "operand_precision": self.operand_precision,
            "core_allocation": self.core_allocation,
            "user_spatial_mapping": self.user_spatial_mapping,
            "memory_operand_links": self.memory_operand_links,
            "source_storage_level": self.source_storage_level
        }

    def calc_tensor_size(self, layer_op, loop_sizes):  # layer_op 只能是'O', 'W', 'I'
        """
        Calculates the tensor size (nb of elements) for the given operand layer_op with the given loop dimension sizes loop_sizes.
        :param layer_op: str. A String representing the layer operand for which to compute the tensor size.
        :param loop_sizes: dict. A dict with string keys representing the dimension and integer values representing the size.
        """
        return prod(ceil(va) for va in self.calc_tensor_dims(layer_op, loop_sizes).values())
        # Initialize the tensor size as 1

    def calc_tensor_dim(self, layer_op: str, loop_sizes: dict, dim: str):  # 用于计算 IFM 的维度，是从OFM 和等式关系中计算出来的
        if dim in loop_sizes:
            return loop_sizes[dim]
        elif dim in self.operand_loop_dim[layer_op]['pr']:
            related_dimension_sizes = [loop_sizes[dimension] for dimension in self.pr_loop[dim]]
            scaling_factors = list(self.pr_scaling_factors[dim].values())
            assert len(related_dimension_sizes) == len(
                scaling_factors) == 2, "Shouldn't happen if partial relevancy checks in extract_pr_loop_info() are done correctly."
            args = (val for pair in zip(scaling_factors, related_dimension_sizes) for val in pair)
            return self.calc_pr_dimension_size(*args)
        else:
            assert False

    def calc_tensor_dim_fraction(self, layer_op: str, loop_sizes: dict, dim: str):  # 用于计算 IFM 的维度，是从OFM 和等式关系中计算出来的
        ''' 这里专门用于combined_mapping中计算 I 的解耦合, 在mapping_assist_funcs.decouple_pr_loop()被调用 '''
        if dim in loop_sizes:
            return loop_sizes[dim]
        elif dim in self.operand_loop_dim[layer_op]['pr']:
            related_dimension_sizes = [loop_sizes[dimension] for dimension in self.pr_loop[dim]]
            scaling_factors = list(self.pr_scaling_factors[dim].values())
            assert len(related_dimension_sizes) == len(
                scaling_factors) == 2, "Shouldn't happen if partial relevancy checks in extract_pr_loop_info() are done correctly."
            args = (val for pair in zip(scaling_factors, related_dimension_sizes) for val in pair)
            return self.calc_pr_dimension_size_fraction(*args)
        else:
            assert False

    def calc_tensor_dims(self, layer_op: str, loop_sizes: dict):
        # Ceil the loop sizes so that the size calculation will be correct in case of fractional spatial loop

        # 把loop size的小数变成整数
        loop_sizes = {key: ceil(val) for key, val in loop_sizes.items()}
        out = {}
        op_dimensions = self.operand_loop_dim[layer_op]
        for dim in op_dimensions['r'] + list(op_dimensions['pr'].keys()):
            out[dim] = self.calc_tensor_dim(layer_op, loop_sizes, dim)
        return out

    @staticmethod
    def calc_pr_dimension_size(sa, A, sb, B):
        """
        Calculates the number of unique indices c generated by iterating through the indices
        a in range(0,A,1) and b in range(0,B,1) according to the equation c = sa * a + sb * b.
        sa and sb thus represent the scaling of a, resp. b.
        """
        return int(A * B - max(0, B - (sa / gcd(sa, sb))) * (A - (sb / gcd(sa, sb))))

    @staticmethod
    def calc_pr_dimension_size_fraction(sa, A, sb, B):
        return (A * B - max(0, B - (sa / gcd(sa, sb))) * (A - (sb / gcd(sa, sb))))

    @staticmethod
    def return_lambda(equal_sign_right):
        return eval("lambda n: " + equal_sign_right)

    @staticmethod
    def extract_pr_loop_info(equation_relations):
        pr_loop: Dict[str, list] = {}
        pr_loop_list: List[str] = []
        pr_scaling_factors: Dict[str, list] = {}
        for relation in equation_relations:
            relation_disassembly = re.findall('[a-zA-Z]+', relation)

            assert len(
                relation_disassembly) == 3, f"equation_relation {relation} does not involve a linear relationship between two dimension iterators."

            key = relation_disassembly[0].upper()
            val = [loop_dim.upper() for loop_dim in relation_disassembly[1:]]
            pr_loop[key] = val
            pr_loop_list.extend([key] + val)

            # To extract the scaling factors for the different loop dimension iterators, we need to make sure
            # there is a scaling factor present in the equation. If it is not present, raise an exception.
            scaling_factors = {}
            for val_lower in relation_disassembly[1:]:
                if relation[relation.index(val_lower) - 1] == '*':
                    if not relation[relation.index(val_lower) - 2].isdigit():
                        raise NotImplementedError(f"Please use a scaling factor for every dimension iterator on the RHS of equation {relation}")
                    else:
                        scaling_factors[val_lower] = int(re.findall('(\\d+)(?=\\*' + val_lower + ')', relation)[0])
                else:
                    scaling_factors[val_lower] = 1
            #scaling_factors = re.findall('[0-9]+', relation)
            assert len(scaling_factors) == 2, f"Please remove any constants in the equation relation {relation}."
            pr_scaling_factors[key] = scaling_factors
        return pr_loop, pr_loop_list, pr_scaling_factors

    @staticmethod
    def extract_r_ir_loop_info(equation, loop_dim_size, pr_loop, pr_loop_list):
        '''解析 equation'''
        operand_loop_dim: Dict[str, Dict] = {}
        operand_list = []
        equation = equation.replace('*', ' * ')
        equation = equation.replace('=', ' = ')
        equation = equation.replace('+', ' + ')
        equation = equation.replace('-', ' - ')
        equation_disassembly = re.findall('[a-zA-Z,=,*,+,-]+', equation)
        # filter out + that directly precedes an = (+=) or another + (++) to make this work for concat and add
        prev_char = None
        for i, char in enumerate(equation_disassembly):
            if (char == '=' or char == '+') and prev_char == '+':
                equation_disassembly.pop(i - 1)
            prev_char = char
        split_location = [i for (i, x) in enumerate(equation_disassembly) if x in ['=', '*', '+', '-']] + [len(equation_disassembly)]
        dimension_list = list(loop_dim_size.keys())
        begin_idx = 0
        for split_loc in split_location:
            operand = equation_disassembly[begin_idx]
            operand_list.append(operand)
            operand_loop_dim[operand] = {}
            r_loop_list = [loop_dim.upper() for loop_dim in equation_disassembly[begin_idx + 1:split_loc]]
            ir_loop_list = list(set(dimension_list).difference(r_loop_list))

            pr_loop_remove_flag = any(loop in list(pr_loop.keys()) for loop in r_loop_list)
            if pr_loop_remove_flag:
                operand_loop_dim[operand]['r'] = [loop for loop in r_loop_list if loop not in pr_loop_list and loop_dim_size[loop] != 1]
                operand_loop_dim[operand]['ir'] = [loop for loop in ir_loop_list if loop not in pr_loop_list and loop_dim_size[loop] != 1]
                operand_loop_dim[operand]['pr'] = pr_loop
            else:
                operand_loop_dim[operand]['r'] = [loop for loop in r_loop_list if loop_dim_size[loop] != 1]
                operand_loop_dim[operand]['ir'] = [loop for loop in ir_loop_list if loop_dim_size[loop] != 1]
                operand_loop_dim[operand]['pr'] = {}
            begin_idx = split_loc + 1
        ''' operand_loop_dim_reform remove the pr loop dict, and put the pr-related data dimension (e.g. IX and IY)
         to r and ir dict with "_r" and "_ir" suffix. It brings benefits to loop info extraction after pr loop decoupling step. '''
        operand_loop_dim_reform = deepcopy(operand_loop_dim)
        for operand, dic in operand_loop_dim.items():
            del operand_loop_dim_reform[operand]['pr']
            if dic['pr'] != {}:
                r_extend_list = [pr_data_dim + '_r' for pr_data_dim in pr_loop.keys()]
                ir_extend_list = [pr_data_dim + '_ir' for pr_data_dim in pr_loop.keys()]
                operand_loop_dim_reform[operand]['r'] += r_extend_list
                operand_loop_dim_reform[operand]['ir'] += ir_extend_list

        return operand_loop_dim, operand_loop_dim_reform, operand_list

    def extract_layer_info(self):
        """
        This function extract basic information for each layer node.
        :return: total_MAC_count, operand_size_elem, operand_size_bit, operand_data_reuse. 
        :总的MAC数: int、    \\
        :每个op的size: Dict[str, int]、  \\
        :每个op的bit size: Dict[str, int]、  \\
        :每个操作数的总数据重用因子, 即总MAC操作数/总操作数大小: Dict[str, float]
        """
        ''' total MAC operation count '''
        total_MAC_count: int = 1
        for ky in self.loop_dim_size:
            total_MAC_count *= self.loop_dim_size[ky]
        self.total_MAC_count = total_MAC_count
        ''' each operand's size (Unit: # of data element) '''
        operand_size_elem: Dict[str, int] = {}
        for operand, relevancy in self.operand_loop_dim.items():
            operand_size_elem[operand] = 1
            for r_loop in relevancy['r']:
                operand_size_elem[operand] *= self.loop_dim_size[r_loop]
            for pr_loop, pr_loop_collect in relevancy['pr'].items():
                multiply_factor = self.calc_tensor_dims(operand, self.loop_dim_size)[pr_loop]
                operand_size_elem[operand] *= multiply_factor
        self.operand_size_elem = operand_size_elem
        ''' each operand's size (Unit: bit) '''
        operand_size_bit: Dict[str, int] = {}
        for operand, size_in_elem in operand_size_elem.items():
            operand_size_bit[operand] = size_in_elem * self.operand_precision[operand]
        self.operand_size_bit = operand_size_bit
        ''' each operand's total data reuse factor, which is total MAC Op/total operand size (in element), 
        每个操作数的总数据重用因子, 即总MAC操作数/总操作数大小
        i.e. each data element can be used to support how many MAC operation. '''
        operand_data_reuse: Dict[str, float] = {}
        for operand, size_in_elem in operand_size_elem.items():
            operand_data_reuse[operand] = total_MAC_count / size_in_elem
        self.operand_data_reuse = operand_data_reuse

    def get_operand_irrelevant_dimensions(self, layer_op: str) -> List[str]:
        """
        返回和给定op不相关的维度op, 如 O 的不相关为 ['FX', 'FY', 'C'] \\
        Return the irrelevant dimensions of layer operand 'layer_op'.
        """
        return self.operand_loop_dim[layer_op]['ir']

    def get_layer_operand(self, mem_op: str) -> str:
        """
        给一个内存表示的op, 三种情况: 'O', 'I1', 'I2', 返回对应的layer中的op, 'O', 'I', 'W'. \\
        Return the layer operand associated with the given memory operand for this layer.
        If there is no such memory operand, an error is raised.
        """
        for layer_operand, memory_operand in self.memory_operand_links.items():
            if memory_operand == mem_op:
                return layer_operand
        raise ValueError(f"The memory operand {mem_op} is not present in layer {self}.")

    def get_operand_storage_level(self, layer_op: str):
        """
        Return the memory level at which an input operand is stored.
        If this layer node has no information for the given operand, it returns None.
        """
        if layer_op not in self.source_storage_level:
            return None
        return self.source_storage_level[layer_op]


if __name__ == "__main__":

    ld_c = {
        'operator_type': 'Conv',  # kernel 32x3x3x3,  output 1x32x112x112
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': { 'B': 1, 'K': 32, 'OX': 112, 'OY': 112, 'C': 3, 'FX': 3, 'FY': 3 },
        'operand_precision': { 'O': 16, 'O_final': 8, 'W': 8, 'I': 8 },
        'operand_source': { 'W': [], 'I': [-1] },
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': { 'I': { 'IX': 'OX', 'IY': 'OY', 'C': 'K' } },
        'core_allocation': 1,
        'spatial_mapping': { 'D1': ('K', 32), 'D2': ('C', 32) },
        'memory_operand_links': { 'O': 'O', 'W': 'I2', 'I': 'I1' }
    }

    ld_t = {
        'operator_type': 'Transpose',  # 在网络描述中，假设什么都不做，1x32x112x112 -> 1x32x112x112，但是在后层执行中计算data_copy的数据量  
        'equation': 'transpose',
        'equation_relations': [],
        'loop_dim_size': { 'B': 1, 'G': 32, 'OX': 112, 'OY': 112 },
        'operand_precision': { 'O': 8, 'O_final': 8, 'I': 8 },
        'operand_source': { 'I': [0] },
        'constant_operands': [],
        'operand_source_dimension_mapping': { 'I': { 'OX': 'OX', 'OY': 'OY', 'G': 'K' } },
        'core_allocation': 1,
        'spatial_mapping': {},
        'memory_operand_links': { 'O': 'O', 'I': 'I1' }
    }

    ld_p = {  # max pool, stride 2
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': { 'B': 1, 'G': 64, 'OY': 155, 'OX': 155, 'FX': 3, 'FY': 3 },
        'operand_precision': { 'O': 16, 'O_final': 8, 'I': 8, 'W': 0 },
        'operand_source': { 'W': [], 'I': [0] },
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': { 'I': { 'IX': 'OX', 'IY': 'OY', 'G': 'K' } },
        'core_allocation': 1,
        'spatial_mapping': { 'D1': ('G', 32) },
        'memory_operand_links': { 'O': 'O', 'I': 'I1', 'W': 'I2' }
    }

    ld_a = {  # Addition of layer 13 (residual connection) and layer 15 (main path)
        'equation': 'O[b][g][oy][ox]=X[b][g][oy][ox]+Y[b][g][oy][ox]',
        'equation_relations': [],
        'loop_dim_size': { 'B': 1, 'G': 256, 'OY': 31, 'OX': 31 },
        'operand_precision': { 'O': 16, 'O_final': 8, 'X': 8, 'Y': 8 },
        'operand_source': { 'X': [13], 'Y': [15] },
        'constant_operands': [],
        'operand_source_dimension_mapping': { 'X': { 'OX': 'OX', 'OY': 'OY', 'G': 'G' }, 'Y': { 'OX': 'OX', 'OY': 'OY', 'G': 'K' } },
        'core_allocation': 1,
        'spatial_mapping': { 'D1': ('G', 32) },
        'memory_operand_links': { 'O': 'O', 'X': 'I1', 'Y': 'I1' }
    }

    # operand_loop_dim:
    # {
    # 'O': {'r': ['K', 'OY', 'OX'],         'ir': ['FY', 'C', 'FX'],    'pr': {}},
    # 'W': {'r': ['K', 'C', 'FY', 'FX'],    'ir': ['OX', 'OY'],         'pr': {}},
    # 'I': {'r': ['C'],                     'ir': ['K'],                'pr': {'IX': ['OX', 'FX'], 'IY': ['OY', 'FY']}}
    # }

    # operand_loop_dim_reform:
    # {
    # 'O': {'r': ['K', 'OY', 'OX'],         'ir': ['FY', 'C', 'FX']},
    # 'W': {'r': ['K', 'C', 'FY', 'FX'],    'ir': ['OX', 'OY']},
    # 'I': {'r': ['C', 'IX_r', 'IY_r'],     'ir': ['K', 'IX_ir', 'IY_ir']}
    # }

    # operand_list:  ['O', 'W', 'I']         或者add的情况：['O', 'X', 'Y']
    # pr_scaling_factors：  {'IX': {'ox': 2, 'fx': 1}, 'IY': {'oy': 2, 'fy': 1}}    这是从equation_relation中提取的

    ld = ld_p
    nd = LayerNode(1, ld)
    # aa = nd.build_pr_funcs()
    # print(aa)
    print('---' * 30)

    print('pr_scaling_factors: ', nd.pr_scaling_factors)  # {'IX': {'ox': 2, 'fx': 1}, 'IY': {'oy': 2, 'fy': 1}}
    print('pr_loop: ', nd.pr_loop)  # {'IX': ['OX', 'FX'], 'IY': ['OY', 'FY']}
    print('pr_loop_list: ', nd.pr_loop_list)  # ['IX', 'OX', 'FX', 'IY', 'OY', 'FY']
    print('---' * 30)

    print('operand_loop_dim:\n ', nd.operand_loop_dim)
    print('operand_loop_dim_reform:\n ', nd.operand_loop_dim_reform)
    print('operand_list:\n ', nd.operand_list)
    print('---' * 30)

    s_1 = nd.calc_tensor_size('O', ld['loop_dim_size'])  # 输出的是 OFM 的大小，即 OX * OY * K
    s_2 = nd.calc_tensor_dims(
        'W',
        ld['loop_dim_size'])  # 根据字符'I'输出 {'C': 3, 'IX': 225, 'IY': 225}，是 IFM 的维度信息，是根据输出特征和卷积核推理计算出来的. 'W'则输出{'K': 32, 'C': 3, 'FY': 3, 'FX': 3}
    s_3 = nd.calc_tensor_dim('I', ld['loop_dim_size'], 'IY')  # 用于计算 IFM 的维度
    print('tensor size of O: ', s_1)
    print('tensor dims of W: ', s_2)
    print('tensor dim of I:  ', s_3)
    print('---' * 30)

    ll = nd.get_operand_irrelevant_dimensions('O')
    print('irrelevant_dimensions:', ll)
    print('---' * 30)

    print(
        nd.total_MAC_count
    )  # 10838016                                              # 这是把loop_dim_size中所有维度的大小乘起来了 ('B': 1, 'K': 32, 'OX': 112, 'OY': 112, 'C': 3, 'FX': 3, 'FY': 3)
    print(nd.operand_size_elem)  # {'O': 401408, 'W': 864, 'I': 151875}                  # 这是OFM、kernel、IFM的数据量大小
    print(nd.operand_size_bit)  # {'O': 6422528, 'W': 6912, 'I': 1215000}               # O 是16倍的关系，I 和W 是8倍关系
    print(nd.operand_data_reuse)  # {'O': 27.0, 'W': 12544.0, 'I': 71.36142222222222}     # 这是对应的op，做: total_MAC_count/operand_size_elem
