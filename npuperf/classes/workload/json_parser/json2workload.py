import importlib
import json
import logging
import os
from pprint import pprint
from typing import Any, Dict, List

from npuperf.classes.workload.json_parser.AddMul_parser import (AddParser, MulParser, SubtractParser)
from npuperf.classes.workload.json_parser.Conv_parser import (ConvParser, DeConvParser, DenseParser, dwConvParser)
from npuperf.classes.workload.json_parser.Matmul_parser import MatmulParser
from npuperf.classes.workload.json_parser.MemOp_parser import (ConcatParser, TransposeParser, SplitParser)
from npuperf.classes.workload.json_parser.Pool_parser import PoolParser
from npuperf.classes.workload.json_parser.parser import Parser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Json2WorkloadParser:

    def __init__(self, json_workload_or_path, mapping_path_or_dict, merge_activation_function):
        self.workload_file = {}  # 我们最终得到的 workload file 描述文件
        self.all_node = {}  # Json 描述的整个网络，格式为 id: {'in': str, 'op': str, 'out': str}，在update_operand_source_dict 中会添加 'real_id' 项
        self.mapping = self.pick_mapping(mapping_path_or_dict)
        self.layer_li, self.input_names = self.pick_json_workload(json_workload_or_path)
        self.set_dummy_operator(merge_activation_function)

    def pick_mapping(self, mapping_path_or_dict) -> Dict[str, Dict]:
        if isinstance(mapping_path_or_dict, str):  # path
            module = importlib.import_module(mapping_path_or_dict)
            mapping_path_or_dict = module.mapping  # 取出mapping dict
        # mapping dict from Mapping Generator Stage
        if not isinstance(mapping_path_or_dict, dict):
            raise TypeError(f'mapping is neither dict nor str, type is {type(mapping_path_or_dict)}')
        return mapping_path_or_dict

    def pick_json_workload(self, json_workload_or_path):
        if isinstance(json_workload_or_path, str):
            with open(json_workload_or_path) as f:
                json_workload_or_path: Dict[str, Any] = json.load(f)
        if not isinstance(json_workload_or_path, dict):
            raise TypeError(f'"json_workload_or_path" is not a json_path or json_dict, type is {type(json_workload_or_path)}')

        # layer_li 是一个List，里面的每一个Dict是一个网络层
        layer_li: List[Dict] = json_workload_or_path['layers']
        # 整个网络输入层的名字, 可能存在多输入的网络（例如Bert small是三输入的）
        input_names: list[str] = json_workload_or_path['input_names']
        return layer_li, input_names

    def set_dummy_operator(self, merge_activation_function):
        # 在解析时应当 删去 的网络层，删去并给出warning，被删掉的算子被认为没有加速空间，或没有硬件设计的探索空间
        dummy_op = set({
            'qnn.csi.lrn', 'qnn.csi.softmax', 'qnn.csi.reshape', 'qnn.csi.clip', 'qnn.csi.sigmoid', 'qnn.csi.mean', 'qnn.csi.tanh',
            'qnn.csi.power', 'qnn.csi.sqrt', 'qnn.csi.div', 'qnn.csi.take', 'qnn.csi.erf', 'qnn.csi.strided_slice',
            'qnn.csi.variance', 'qnn.csi.cast', 'qnn.csi.sin', 'qnn.csi.cos', 'qnn.csi.upsampling', 'qnn.csi.exp',
        })
        if not merge_activation_function:  # 如果不融合激活函数，就把非线性放进dummy op里，解析时删去即可
            dummy_op.update({'qnn.csi.relu', 'qnn.csi.prelu'})
        self.dummy_op = dummy_op

    def run(self):
        self.parse_input_layer()
        self.get_io_name()
        self.parse_layer()
        self.update_operand_source_dict()
        return self.workload_file  # .py format workload file we want

    def parse_input_layer_old(self):
        # 先解析InputLayerNode层，即 -1 层
        input_layer_dict = {}
        for input_dict in self.layer_li[0]['inputs']:
            if input_dict['name'] == self.input_names:
                input_layer_dict['equation'] = 'input'
                if input_dict['layout'] == "NCHW":
                    input_layer_dict['loop_dim_size'] = {
                        'B': input_dict['dim'][0],
                        'K': input_dict['dim'][1],
                        'OY': input_dict['dim'][2],
                        'OX': input_dict['dim'][3]
                    }
                else:
                    raise ValueError(f'dim of {input_dict["dim"]} is Not NCHW format! It is {input_dict["layout"]} format!')
                input_layer_dict['precision'] = 8
                input_layer_dict['core_allocation'] = self.mapping['Input']['core_allocation']
                input_layer_dict['memory_operand_links'] = self.mapping['Input']['memory_operand_links']
                break
            else:
                raise ValueError('The first layer item of "layers" in Json file DO NOT find input_name.')
        self.all_node[-1] = {'in': None, 'op': 'INPUT', 'out': [self.input_names]}
        self.workload_file[-1] = input_layer_dict
    
    def parse_input_layer(self):

        def gen_loop_dim_size(input_di: dict):
            assert not input_di['is_const'], f'input dict is not a feature map! {input_di}'
            return {(Parser.DIM_LINK[that_format]).upper(): dim for that_format, dim in zip(list(input_di['layout']), input_di['dim'])}

        # 先解析InputLayerNode层
        for layer_id, input_name in enumerate(self.input_names, start=-1*len(self.input_names)):
            # 对于多输入的网络，有多个 InputLayerNode
            input_layer_dict = {}
            for layer in self.layer_li:
                for in_di in layer['inputs']:
                    if in_di['name'] == input_name:
                        input_layer_dict['equation'] = 'input'
                        input_layer_dict['loop_dim_size'] = gen_loop_dim_size(in_di)
                        input_layer_dict['precision'] = 8
                        input_layer_dict['core_allocation'] = self.mapping['Input']['core_allocation']
                        input_layer_dict['memory_operand_links'] = self.mapping['Input']['memory_operand_links']
            self.all_node[layer_id] = {'in': None, 'op': 'INPUT', 'out': [input_name]}
            self.workload_file[layer_id] = input_layer_dict

        assert len(self.all_node) == len(self.workload_file) == len(self.input_names), f'input layer parse ERROR, all_node={self.all_node}'

    def get_io_name(self):
        i = 0
        deconv_operator_count = 0
        for layer_di in self.layer_li:
            op = layer_di['op_type']
            in_name = [in_di['name'] for in_di in layer_di['inputs'] if in_di['is_const'] == 0]
            out_name = [out_di['name'] for out_di in layer_di['outputs'] if out_di['is_const'] == 0]
            if op == "qnn.csi.deconv2d":  # 从 1 开始计数
                deconv_operator_count += 1
                y_stride, x_stride = layer_di['attrs']['strides']
                new_layer_count = x_stride * y_stride  # 一般是 4
                while new_layer_count > 0:
                    self.all_node[i] = {'in': in_name, 'op': 'Conv_form_deconv', 'out': [f'contract_{deconv_operator_count}_{new_layer_count}']}
                    i += 1
                    new_layer_count -= 1
                self.all_node[i] = {
                    'in': [f'contract_{deconv_operator_count}_{index}' for index in range(1, x_stride * y_stride + 1)],
                    'op': 'Contract',
                    'out': out_name
                }
                i += 1

            else:
                self.all_node[i] = {'in': in_name, 'op': op, 'out': out_name}
                i += 1
        logger.info(f'Json network has {len(self.all_node) - 1} layers.')

    def parse_layer(self):
        # 然后解析网络层，逐层解析，如果遇见激活函数，那就在上一层的dict中加一条 'activation function'，如果遇见不能解析的，如 reshape和softmax，那就删掉并给warning
        id = 0
        # 在解析时实际被删掉或fuse的算子层，可能有relu, prelu, reshape, softmax, LRN 等等
        being_del = set([])
        deconv_count = 0
        for layer in self.layer_li:
            op_type: str = layer['op_type']

            if op_type in self.dummy_op:
                being_del.add(op_type)
                # logger.warning(f'The operator {op_type} is being deleted in parse stage.')
            elif op_type == "qnn.csi.relu":  # 只有在需要合并激活函数的时候，才会执行。因为不合并的话，这个op已经存放在dummy_op中
                being_del.add(op_type)
                self.workload_file[id - 1]['activation_function'] = 'relu'
            elif op_type == "qnn.csi.prelu":  # 只有在需要合并激活函数的时候，才会执行。因为不合并的话，这个op已经存放在dummy_op中
                being_del.add(op_type)
                self.workload_file[id - 1]['activation_function'] = 'prelu'
            else:
                if op_type == "qnn.csi.conv2d" and layer["inputs"][1]["layout"] == "O1HW":
                    parser = dwConvParser(layer, self.mapping)
                elif op_type == "qnn.csi.deconv2d":
                    deconv_count += 1
                    parser = DeConvParser(layer, self.mapping, deconv_count)
                elif op_type == "qnn.csi.conv2d":  # CONV 一定要放在 dw conv 和 deconv 后面，否则会产生误认
                    parser = ConvParser(layer, self.mapping)
                elif op_type in ("qnn.csi.avgpool2d", "qnn.csi.global_avgpool2d", "qnn.csi.global_maxpool2d", "qnn.csi.maxpool2d"):
                    parser = PoolParser(layer, self.mapping)
                elif op_type == "qnn.csi.add":
                    parser = AddParser(layer, self.mapping)
                elif op_type == "qnn.csi.subtract":
                    parser = SubtractParser(layer, self.mapping)
                elif op_type == "qnn.csi.mul":
                    parser = MulParser(layer, self.mapping)
                elif op_type == "qnn.csi.dense":
                    parser = DenseParser(layer, self.mapping)
                elif op_type == "qnn.csi.matmul":
                    parser = MatmulParser(layer, self.mapping)
                elif op_type == "qnn.csi.concatenate":
                    parser = ConcatParser(layer, self.mapping)
                elif op_type == "qnn.csi.transpose":
                    parser = TransposeParser(layer, self.mapping)
                elif op_type == "qnn.csi.split":
                    parser = SplitParser(layer, self.mapping)
                else:
                    raise NameError(f'The operator type {op_type} unsupported.')
                layer_dict_raw = parser.run()

                if op_type == "qnn.csi.deconv2d":
                    # 如果是deconv，那返回的layer_dict 并不是一层，而是若干层卷积和一层contract，由一个大的dict 包起来
                    for deconv_di in layer_dict_raw.values():
                        layer_dict = parser.add_layer_name_from_json(deconv_di)
                        self.workload_file[id] = layer_dict
                        id += 1
                    # 一个 deconv 层解析出来应该是 x_stride * y_stride 个卷积层 + 1个contract 层
                    logger.info(f'Parsed {op_type:25} layer at id {id - len(layer_dict)} -- {id - 1}')
                else:
                    layer_dict = parser.add_layer_name_from_json(layer_dict_raw)
                    self.workload_file[id] = layer_dict
                    logger.info(f'Parsed {op_type:25} layer at id {id}')
                    id += 1
        self.being_del = being_del
        logger.warning(f'These operators are being deleted in parse stage: {self.being_del}')
        logger.info(f'Parse DONE, the workload has {id} layers.')

    def update_operand_source_dict(self):
        """ 使用 workload_file , being del 和 all_node, 更新operand_source 条目, 从name表达改为需要的id表达 """
        # 先更新 self.all_node, 之后用这个更新 workload file
        real_id = -1 * len(self.input_names)
        for node in self.all_node.values():
            # 先在self.all_node 中添加 real_id 项
            if node['op'] in self.being_del:
                node['real_id'] = None
            else:
                node['real_id'] = real_id
                real_id += 1

            # 然后对当前node, 'in'中的每一个str，在前面的 'out' 找对应的str，并修改in 为id值。两种情况：如果有real_id, 那么改为real_id；如果没有real_id，那么改为那个nd的in
            if node['op'] == 'INPUT':
                continue
            for i, in_n in enumerate(node['in']):
                if type(in_n) is int:  # 如果in_n已经是数字，说明已经改掉了，那就跳过
                    continue
                node['in'][i] = self.__change_input(in_n)

        # 最后更改 self.workload_file 中的operand_source 条目
        for id, layer in self.workload_file.items():
            if layer['equation'] == 'input':
                continue
            elif layer['operator_type'] in ['Add', 'Mul', 'Matmul', 'Subtract']:
                if layer['operand_source'].get('I', None):
                    layer['operand_source']['I']: list = self.__get_source_id(id)
                elif layer['operand_source'].get('X') and layer['operand_source'].get('Y'):
                    layer['operand_source']['X']: list = [self.__get_source_id(id)[0]]
                    layer['operand_source']['Y']: list = [self.__get_source_id(id)[1]]
                else:
                    raise KeyError(f'Unexpected layer at id={id}: {layer}')

            elif layer['operator_type'] in ['Concat', 'Contract']:
                source_id = self.__get_source_id(id)
                for i in range(len(layer['operand_source'])):
                    layer['operand_source'][f'X{i}'] = [source_id[i]]
            else:
                layer['operand_source']['I'] = self.__get_source_id(id)


    def __change_input(self, in_n: str) -> int:
        """ 给定input name, 返回这个输出是这个name的那一层的real_id, 如果没有real_id, 那么改为那个node的in """
        for nd in self.all_node.values():
            if any(out_n == in_n for out_n in nd['out']):
                # find output name str
                if nd['real_id'] == None:
                    in_int = nd['in'][0]
                else:
                    in_int = nd['real_id']
                return in_int

    def __get_source_id(self, id) -> List[int]:
        """ 给定当前layer 的id, 返回source 的id """
        for node in self.all_node.values():
            if node['real_id'] == id:
                return node['in']

    def export(self, export_path):
        """export .py format workload dict to file at export_path
        """
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        with open(export_path, 'w') as ff:
            print('workload =', file=ff, end='')
            pprint(object=self.workload_file, stream=ff, indent=4, width=150, sort_dicts=False)
        # with open('all_node.py', 'w') as fi:
        #     pprint(object=self.all_node, stream=fi, indent=4, width=150, sort_dicts=False)
        logger.info(f'NetWork File export DONE!, at path {export_path} \n' + '---' * 60)


if __name__ == '__main__':

    mapping_fold_path = f'inputs/Mapping'
    mapping_path_list = os.listdir(mapping_fold_path)  # os.listdir 列出文件夹中的文件
    mapping_path_list.remove('__pycache__')
    # DLA_list = ['Meta_prototype','Ascend_like','Edge_TPU_like','Tesla_NPU_like','TPU_like']
    # DLA = 'Meta_prototype'
    # DLA = 'Ascend_like'
    # DLA = 'Edge_TPU_like'
    # DLA = 'Tesla_NPU_like'
    # DLA = 'TPU_like'

    # NN = 'fcn8s'
    # NN = 'fsrcnn2x'
    # NN = 'inceptionv1'
    # NN = 'resnet50'
    # NN = 'mv1'
    NN = 'mv2'

    merge_activation_function = False

    WL_folder = "WL_fromjson" if merge_activation_function else "WL_fromjson_wioAct"
    json_workload_path = f'inputs/hhb_networks/{NN}.json'  # = 网络描述json 文件

    # = MODE 1   一个网络迭代所有DLA
    # for DLA in mapping_path_list:
    #     DLA_name = DLA.split('.')[0]
    #     export_path = f"inputs/{WL_folder}/{DLA_name}/workload_{NN}.py"
    #     mapping_path = f'inputs.Mapping.{DLA_name}'      #= 不同的DLA 有不同的mapping
    #     logger.info(f'Parse network {NN} for Current DLA: {DLA_name}')

    #     json_parser = Json2WorkloadParser(json_workload_path, mapping_path, export_path, merge_activation_function)
    #     json_parser.run()

    # = MODE 2   一个网络，一个DLA
    DLA_name = 'Meta_prototype'
    export_path = f"inputs/{WL_folder}/{DLA_name}/workload_{NN}.py"
    mapping_path = f'inputs.Mapping.{DLA_name}'  # = 不同的DLA 有不同的mapping
    logger.info(f'Parse network {NN} for Current DLA: {DLA_name}')
    json_parser = Json2WorkloadParser(json_workload_path, mapping_path, export_path, merge_activation_function)
    json_parser.run()

    # visualization
    logger.info('Visualizing network files obtained... Check DNN graph and close it to continue.')
    from classes.workload.dnn_workload import DNNWorkload
    from visualization.graph.dnn import visualize_dnn_graph
    workload = DNNWorkload(json_parser.workload_file)
    visualize_dnn_graph(workload)
