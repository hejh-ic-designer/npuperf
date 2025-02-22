import importlib
import logging
from typing import List, Any

from npuperf.classes.stages.Stage import Stage
from npuperf.classes.workload.dnn_workload import DNNWorkload
from npuperf.classes.workload.json_parser.json2workload import Json2WorkloadParser
from npuperf.classes.opt.hw_gen.hardware_modifier import HardwareModifier
from npuperf.classes.opt.hw_gen.hardware_generator import HardwareGenerator

logger = logging.getLogger(__name__)


def parse_accelerator_from_path(accelerator_path):
    """
    Parse the input accelerator residing in accelerator_path.
    """
    global module
    module = importlib.import_module(accelerator_path)
    accelerator = module.accelerator
    logger.info(f"Parsed accelerator with cores {[core.id for core in accelerator.cores]}.")
    return accelerator


class AcceleratorParserStage(Stage):

    def __init__(self, list_of_callables, *, accelerator_path, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator_path = accelerator_path

    def run(self):
        accelerator = parse_accelerator_from_path(self.accelerator_path)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


def parse_workload_from_path(workload_path) -> DNNWorkload:
    """
    Parse the input workload residing in accelerator_path.
    The "workload" dict is converted to a NetworkX graph.

    可以接受workload path, 也可以接受dict 类型的workload
    """
    if isinstance(workload_path, dict):
        workload = DNNWorkload(workload_path)
    elif isinstance(workload_path, str):
        module = importlib.import_module(workload_path)
        workload = module.workload
        workload = DNNWorkload(workload)
    else:
        raise TypeError(f'workload_path {workload_path} type ERROR, neither dict nor str.')
    logger.info(f"Created workload graph with {workload.number_of_nodes()} nodes and {workload.number_of_edges()} edges.")
    return workload


class WorkloadParserStage(Stage):

    def __init__(self, list_of_callables, *, workload_path, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload_path

    def run(self):
        workload = parse_workload_from_path(self.workload)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

class WorkloadParserAddSMStage(Stage):
    """
    解析.py workload, 同时添加mapping (spatial_mapping, core_allocation, memory_operand_links)
    这个 Stage 之前必须要在SimpleMappingGeneratorStage之后
    """

    def __init__(self, list_of_callables, *, workload_path, json_mapping_path_or_dict, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload_path
        self.json_mapping_path_or_dict = json_mapping_path_or_dict

    def run(self):
        self.workload_addSM = self.add_sm()
        workload = parse_workload_from_path(self.workload_addSM)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def add_sm(self):
        # pick workload
        if isinstance(self.workload, dict):
            workload = self.workload
        elif isinstance(self.workload, str):
            module = importlib.import_module(self.workload)
            workload = module.workload
        
        # pick mapping
        if isinstance(self.json_mapping_path_or_dict, dict):
            mapping = self.json_mapping_path_or_dict
        elif isinstance(self.json_mapping_path_or_dict, str):
            module = importlib.import_module(self.json_mapping_path_or_dict)
            mapping = module.mapping
        
        workload_addSM = workload.copy()
        for layer in workload_addSM.values():
            # 如果layer是mem op则没有sm，如果是其他的则有sm
            op = layer.get('operator_type', None)
            if not mapping.get(op):
                continue
            layer['spatial_mapping'] = mapping[op]["spatial_mapping"]
        return workload_addSM

class WorkloadAndAcceleratorParserStage(Stage):
    """
    Convenience class to parse both the workload and accelerator
    """

    def __init__(self, list_of_callables, *, workload_path, accelerator_path, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload_path = workload_path
        self.accelerator_path = accelerator_path

    def run(self):
        workload = parse_workload_from_path(self.workload_path)
        accelerator = parse_accelerator_from_path(self.accelerator_path)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


def parse_json_workload(json_path_or_NN, mapping_path_or_dict: str | dict, export_wl: bool, merge_activation_function: bool = False) -> dict:
    """
    parse json format workload to DNN workload graph

    输入可以是json_path 也可以是 json; mapping 可以是 path, 也可以是mapping dict
    """
    parser = Json2WorkloadParser(json_path_or_NN, mapping_path_or_dict, merge_activation_function)
    workload_dict = parser.run()

    if export_wl:  # 导出 .py format workload to file at export_path
        WL_folder = "WL_fromjson_act" if merge_activation_function else "WL_fromjson"
        DLA_name = mp.split('.')[-1] if isinstance(mp := mapping_path_or_dict, str) else mp['DLA_name']
        wl_name = json_path_or_NN.split('/')[-1][:-5] if isinstance(json_path_or_NN, str) else 'json_network'
        export_path = f"npuperf/inputs/{WL_folder}/{DLA_name}/workload_{wl_name}.py"
        parser.export(export_path)

    return workload_dict


class JsonWorkloadParserStage(Stage):
    """
    parse json format workload from HHB tools to workload.py Dict object
    """

    def __init__(self, list_of_callables, *, json_NN, json_mapping_path_or_dict, export_wl=False, merge_activation_function=False, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.json_NN = json_NN
        self.mapping_path_or_dict = json_mapping_path_or_dict
        self.export_wl = export_wl
        self.merge_activation_function = merge_activation_function

    def run(self):
        self.workload_dict = parse_json_workload(self.json_NN, self.mapping_path_or_dict, self.export_wl, self.merge_activation_function)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload_path=self.workload_dict, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


class HardwareModifierStage(Stage):
    """
    modify accelerator object if user input some hardware config information
    """

    def __init__(self, list_of_callables, args, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.args = args
        # 使用用户提供的config 信息（gb_size, dram_bw, etc.）去改变 base_hw 相对应的值
        self.base_hw = parse_accelerator_from_path(f'npuperf.inputs.HW.{args.hw}')

    def run(self):
        if self.check_select_an_exist_hw():
            accelerator = self.base_hw
        else:
            accelerator = HardwareModifier.from_arg(acc=self.base_hw, config_info=self.args).get_accelerator()
        self.accelerator = accelerator
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=self.accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def check_select_an_exist_hw(self):
        # arg_list is like: [('nn', 'fsrcnn'), ('hw', 'example_wioGB'), ('gb_size', '3'), ('gb_bw', None), ('dram_bw', '512')]
        arg_list: List[tuple[str, Any]] = self.args._get_kwargs()
        # 如果除了前两项，其他都是None，说明没有输入任何的配置信息，则 return True，否则说明存在传入的配置信息，return False
        for (_, arg_v) in arg_list[2:]:
            if arg_v is not None:
                return False
        return True


class HardwareGeneratorStage(Stage):
    """
    generate Accelerator object from a user-defined json format hardware-config file
    """
    def __init__(self, list_of_callables, json_HW: str | dict, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = HardwareGenerator(json_HW).get_accelerator()

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=self.accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
