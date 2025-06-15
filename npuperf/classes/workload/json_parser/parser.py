from typing import Dict, List, Any
from math import ceil


class Parser:
    """
    每一个算子的解析, 都有一个共同点, 就是OFM 的shape 都可以先提取出来 \\
    所以这里会提取 B, K, OY, OX 维度, 在具体的每个算子的parser 里会继承这些信息
    :param: dataprecision: data precision(bit), assume 8 bit, intermidiate output precision is 16
    :inputs_li: input list for json NN layer
    :attrs_di: attrs dict for json NN layer
    :outputs_li: output list for json NN layer
    """
    #这个映射都是相对输出而言，C其实是OC，如何映射看extract_tensor_size
    DIM_LINK = {
        'N': 'b',
        'C': 'g',
        'D': 'd',
        'H': 'oy',
        'W': 'ox',
    }

    def __init__(self, layer: Dict[str, Any], mapping: Dict[str, Any]):
        self.layer = layer
        self.data_precision = 8  # data precision, assume 8 bit, intermidiate output precision is 16 by default
        self.partial_sum_precision: int = mapping['partial_sum_precision']
        self.layer_name_from_json: str = layer['name']

        self.attrs_di: dict = layer.get('attrs')
        self.inputs_li: List[Dict] = layer['inputs']
        self.outputs_li: List[Dict] = layer['outputs']

        if layer['op_type'] == "qnn.csi.split":
            assert (len(self.inputs_li) == 1), f"The input of operator 'qnn.csi.split' is more than 1, please check! layer name:{layer['name']}"
            self.extract_tensor_size(self.inputs_li[0], self.layer)
        else:
            assert (len(self.outputs_li) == 1), f"The output is more than 1, please check! op_type:{layer['op_type']}, layer name:{layer['name']}"
            self.extract_tensor_size(self.outputs_li[0], self.layer)

    def run(self):
        raise ImportError("Run function not implemented for runnable")

    def extract_tensor_size(self, tensor: dict, layer: dict):
        """提取json网络描述中的一个layer中的输出或输入特征图大小

        Args:
            tensor (dict): json中要提取的描述特征图的dict
            layer (dict): json中要提取的这一个layer的dict
        """
        if 'NCHW' == tensor['layout']:
            output_dim_B, output_dim_K, output_dim_OY, output_dim_OX = tensor['dim'][:]
        elif 'NC' == tensor['layout']:  # Dense layer的输出是二维的
            output_dim_B, output_dim_K = tensor['dim'][:]
            output_dim_OY = 1
            output_dim_OX = 1
        elif 'NCW' == tensor['layout']:
            output_dim_B, output_dim_K, output_dim_OX = tensor['dim'][:]
            output_dim_OY = 1
        elif ('NCDHW' == tensor['layout']) and (layer['op_type'] in ["qnn.csi.transpose", "qnn.csi.add", "qnn.csi.mul"]):
            # 目前在能支持的算子中，看到 Add, Mul 和Transpose 有这样的feature
            output_dim_B, output_dim_K, output_dim_D, output_dim_OY, output_dim_OX = tensor['dim'][:]
        elif ('NLCDHW' == tensor['layout']) and (layer['op_type'] in ["qnn.csi.transpose"]):
            # 目前在能支持的算子中，看到 Transpose 有这样的feature
            output_dim_B, output_dim_L, output_dim_K, output_dim_D, output_dim_OY, output_dim_OX = tensor['dim'][:]
        else:
            raise ValueError(
                f'Output Feature Map parse, the OFM dim of {tensor["dim"]} is Not NCHW format! It is {tensor["layout"]} format! op_type: {layer["op_type"]}'
            )

        # 对于NCDHW 和 NLCDHW格式的 Add, Mul, Transpose layer, 设置其 D 和 L 的dim
        try:
            self.output_dim_D = output_dim_D
        except:
            self.output_dim_D = 1
        try:
            self.output_dim_L = output_dim_L
        except:
            self.output_dim_L = 1

        group = self.attrs_di.get('groups', 1) if self.attrs_di else 1  # 卷积中有groups这个参数，不为1的话应该将通道数除以groups，其他算子的话就都设置为 1
        self.output_dim_B = output_dim_B
        self.output_dim_K = ceil(output_dim_K / group)
        self.output_dim_OY = output_dim_OY
        self.output_dim_OX = output_dim_OX

        if layer['op_type'] == "qnn.csi.conv2d" and layer["inputs"][1]["layout"] == "O1HW":  # 对于 dw conv，不用除以 group
            self.output_dim_K = output_dim_K

    @staticmethod
    def get_spatial_mapping_old(op_mapping: Dict[str, Any], loop_dim_size: Dict[str, int]):
        """ 这个根据 loop_dim_size, 如果mapping里的某个值比它大就会选loop里的值 """
        op_spatial_mapping: Dict[str, tuple] = op_mapping.get('spatial_mapping')
        assert op_spatial_mapping, "spatial mapping NOT defined in mapping file!"

        spatial_mapping = {}
        for D, unroll_tuple in op_spatial_mapping.items():
            spatial_mapping[D] = (unroll_tuple[0], min(unroll_tuple[1], loop_dim_size[unroll_tuple[0]]))
            if spatial_mapping[D][1] == 1:  # 如果维度 = 1，那么不能出现在spatial mapping里
                spatial_mapping.pop(D)
        return spatial_mapping

    @staticmethod
    def get_spatial_mapping(op_mapping: Dict[str, Any], loop_dim_size: Dict[str, int]):
        op_spatial_mapping: Dict[str, tuple] = op_mapping.get('spatial_mapping')
        assert op_spatial_mapping, "spatial mapping NOT defined in mapping file!"
        return op_spatial_mapping

    def add_layer_name_from_json(self, layer_dict: Dict[str, Any]):
        """在已经解析好的 layer dict 中添加来自json 网络描述的 name 字段

        Args:
            layer_dict (Dict[str, Any]): 原本.py格式的网络描述的一个层

        Returns:
            Dict[str, Any]: 添加了name字段后的layer dict
        """
        layer_dict.update({'name': self.layer_name_from_json})
        return layer_dict

    @staticmethod
    def _gen_effective_dim_str(tensor_dict: dict):
        loop_dim = list(zip(list(tensor_dict['layout']), tensor_dict['dim']))  # like: [('N', 1), ('C', 12), ('D', 1), ('H', 10), ('W', 10)]
        loop_dim_reform = [i[0] for i in loop_dim if i[1] > 1]
        effective_dim_str = ''.join(['[' + Parser.DIM_LINK[char] + ']' for char in loop_dim_reform])
        return effective_dim_str

    @staticmethod
    def get_equation_WIO(weight_di: dict, operation: str, input_di: dict):
        """当输入张量和权重(常量数据)做加法和点乘时, 权重的数据量和IFM的数据量可能不等
        例如: IFM dim = [1, 12, 14, 14] 而 weight dim = [1, 1, 14, 14] (layout = NCHW)
        这种情况下需要动态配置equation, 维度为 1 的索引不应该出现
        """

        assert weight_di['is_const'], f'Not weight dict! please check your equation configuration, {weight_di}'
        # by gyp 一开始不支持减法，加进入了，原始：['+', '*']
        assert operation in ['+', '*', '-'], f'Unexpected operation: {operation}'

        eff_w_str = Parser._gen_effective_dim_str(weight_di)
        if eff_w_str:
            return f'O[b][g][d][oy][ox]=W{eff_w_str}{operation}I{Parser._gen_effective_dim_str(input_di)}'
        else:
            #这种情况是 权重 的dim为若干个1，例如 [1, 1, 1]
            # 把input dict 的 layout 中不存在的维度任取出一个，在equation中设置，因为这个维度的大小肯定为 1
            dim_not_in_layout = list(set(Parser.DIM_LINK.keys()) - set(list(input_di['layout'])))
            if not dim_not_in_layout:
                # 注意，当weight 的dim = [1, 1, 1, 1, 1] 时，会miss掉，可能需要更多信息判断（例如IFM的dim）
                raise KeyError(f'equation set error, weight dim is {weight_di["dim"]}, input dim is {input_di["dim"]}')
            return f'O[b][g][d][oy][ox]=W[{Parser.DIM_LINK[dim_not_in_layout[0]]}]{operation}I[b][g][d][oy][ox]'

    @staticmethod
    def get_equation_XYO(in_1: dict, operation: str, in_2: dict):
        assert operation in ['+', '*', '-'], f'Unexpected operation: {operation}'
        return f'O[b][g][d][oy][ox]=X{Parser._gen_effective_dim_str(in_1)}{operation}Y{Parser._gen_effective_dim_str(in_2)}'
