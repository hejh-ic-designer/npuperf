import logging
import math
import os
import json
import networkx as nx
from typing import Any, Dict, List, Tuple

from npuperf.classes.hardware.architecture.core import Core
from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from npuperf.classes.hardware.architecture.memory_instance import MemoryInstance
from npuperf.classes.hardware.architecture.memory_level import MemoryLevel
from npuperf.classes.hardware.architecture.operational_array import MultiplierArray
from npuperf.classes.hardware.architecture.operational_unit import Multiplier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CoreGenerator:
    """
    use CoreGenerator.from_dict() method to creat a core generator instance

    use get_core() method to return a core
    
    use save_to_json() method to export hardware config info to a json file at file_path
    """

    def __init__(
        self,
        MAC_unroll: Dict[str, Tuple[str, int]],
        local_buffers: List[Dict[str, str | int]],
        global_buffer: Dict[str, str | int] = None,
        dram: Dict[str, str | int] = None,
        core_id=1,
    ):
        """架构生成器初始化, 建议使用 from_dict() 构造实例

        Args:
            MAC_unroll (dict['dim_id': tuple(op, int)]): 每一个数代表一个维度的展开度
            local_buffer (list[dict['op': W/I/O/IO, 'size': #bits]]): list中的每一个dict表示一块LB, 应当为每个LB 指定其存放的操作数、size
            global_buffer (dict['op': W/I/O/IO, 'size': #bits, 'bandwidth': #bits/cc]): global_buffer应当定义存放的操作数、size、bandwidth
            dram (dict['op': W/I/O/IO, 'size': #bits, 'bandwidth': #bits/cc]): 同 global_buffer 一样
            core_id (int): core id, 1 by default

        Example:
            - MAC_unroll = {'D1': ('K', 4), 'D2': ('C', 16), 'D3': ('OX', 2), 'D4': ('OY', 2)}
            - local_buffers = [
                {'op': 'W', 'size': 16*1024*8},
                {'op': 'I', 'size': 16*1024*8},
                {'op': 'I/O', 'size': 32*1024*8},
            ]
            - global_buffer = {'op': 'I/O', 'size': 0.5*1024*1024*8, 'bandwidth': 128}
            - dram = {'op': 'W', 'size': 4*1024*1024*1024*8, 'bandwidth': 96}       # size is optional
            - core_id = 1
        """
        self.MAC_unroll = MAC_unroll
        self.unroll_counts = len(MAC_unroll)
        self.local_buffers = local_buffers
        self.global_buffer = global_buffer
        self.dram = dram
        self.core_id = core_id

        self.multiplier_array_dut()
        self.reg_instance_dut()
        self.set_served_dimensions()
        self.set_lb_bandwidth()
        self.lb_instance_dut()
        self.gb_instance_dut() if global_buffer else None
        self.dram_instance_dut() if dram else None
        self.memory_hierarchy_dut()
        self.core_dut()

    @classmethod
    def from_dict(cls, core_info: Dict[str, Any]):
        """接受一个dict 模板来创建CoreGenerator 实例

        Args:
            core_info (Dict[str, Any]): 应该包含 MAC_unroll, local_buffers, global_buffer, dram, core_id 等信息

        Example:
            core_info = {
                'MAC_unroll': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 4), 'D4': ('OY', 2)},
                'local_buffers': [
                    { 'op': 'W', 'size': 128 * 1024 * 8 },
                    { 'op': 'I', 'size': 64 * 1024 * 8 },
                    { 'op': 'O', 'size': 128 * 1024 * 8 },
                ],
                'global_buffer': {
                    'op': 'W/I/O',
                    'size': 1 * 1024 * 1024 * 8,
                    'bandwidth': 256
                },
                'dram': {
                    'op': 'W/I/O',
                    'size': 4 * 1024 * 1024 * 1024 * 8,
                    'bandwidth': 192
                },
            }

        Returns:
            CoreGenerator instance
        """
        if cls.check_valid(core_info):

            return cls(
                MAC_unroll=core_info['MAC_unroll'],
                local_buffers=core_info['local_buffers'],
                global_buffer=core_info.get('global_buffer', None),
                dram=core_info.get('dram', None),
                core_id=core_info.get('core_id', 1),
            )

    @classmethod
    def check_valid(cls, core_info: Dict[str, dict]):
        """对输入的模板进行合法性检查, 如果不合法, 则生成 error_dict 抛出
        """
        error_info = {}
        op_check_list = ['I', 'W', 'O', 'I/W', 'I/O', 'W/I', 'W/O', 'O/I', 'O/W', 'I/W/O', 'I/O/W', 'W/I/O', 'W/O/I', 'O/I/W', 'O/W/I']
        try:
            # MAC unroll checking
            for dim_id, unroll in core_info['MAC_unroll'].items():
                if isinstance(dim_id, str) and isinstance(unroll, tuple):  # 首先检查类型
                    if unroll[0] in ['K', 'C', 'OX', 'OY', 'B', 'FX', 'FY'] and isinstance(unroll[1], int):
                        pass
                    else:
                        error_info['MAC_unroll value ERROR'] = f'{unroll} is invalid'
                else:
                    error_info['MAC_unroll type ERROR'] = f'key should be "D1", "D2", "D3" etc. and value should be a tuple'
            # local buffers checking
            for lb in core_info['local_buffers']:
                # for op, size in lb.items():## todo
                if lb['op'] in op_check_list and isinstance(lb['size'], int):
                    pass
                else:
                    error_info['local_buffer ERROR'] = f'operand str: {lb["op"]} error or size type {type(lb["size"])} error'
            # gb and dram checking
            gb = core_info.get('global_buffer', None)
            dram = core_info.get('dram', None)
            if gb and (not gb['op'] in op_check_list):
                error_info['gb operands error'] = f'gb op {gb["op"]} error'
            if dram and (not dram['op'] in op_check_list):
                error_info['dram operands error'] = f'dram op {dram["op"]} error'
            if not (gb or dram):
                error_info['missing gb and dram'] = f'gb and dram should deploy at least one'
        except:
            error_info['NOT implemented ERROR'] = f'core_info missing MAC_unroll or local_buffers'

        if not error_info:
            return True
        else:
            print('core_info:\n', core_info)
            raise TypeError(f'{error_info}')

    def get_core(self) -> Core:
        return self.core1

    def multiplier_array_dut(self):
        """ Multiplier array variables """
        multiplier_input_precision = [8, 8]
        multiplier_energy = 0.04
        multiplier_area = 1

        dimensions = {dim_id: value_tuple[1] for dim_id, value_tuple in self.MAC_unroll.items()}
        # MAC_unroll:  {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)}
        # convert to
        # dimensions:  {'D1': 8, 'D2': 32, 'D3': 2, 'D4': 2}

        multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
        multiplier_array = MultiplierArray(multiplier, dimensions)

        self.dimensions = dimensions
        self.multiplier_array = multiplier_array
        logger.debug(f'multiplier array dimensions: {dimensions}')

    def reg_instance_dut(self):

        # reg 为固定的，不可从外部传参
        #NOTE: 注意 O-reg 的size，bw和 o-reg port energy的关系，这里固定住了size和bw，所以port energy固定为0.03，若为16bit，则应为0.02
        IW_reg_size = 8  # I W reg size = 8 bit
        multiple_factor = 4  # partial_sum_precision_multiple 部分和精度的倍数，如果是 4 说明部分和精度为 32 bit
        O_reg_size = IW_reg_size * multiple_factor  # O reg size = 32 bit
        IW_name = f'rf_{IW_reg_size/8}B'
        O_name = f'rf_{O_reg_size/8}B'

        # reg_IW 是因为I 和 W 的reg 是物理上完全一样的，后面 add_memory 的时候可以添加两次
        reg_IW = MemoryInstance(name=IW_name, size=IW_reg_size, r_bw=IW_reg_size, w_bw=IW_reg_size, r_cost=0.01, w_cost=0.01, area=0, bank=1,
                                 random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)

        reg_O = MemoryInstance(name=O_name, size=O_reg_size, r_bw=O_reg_size, w_bw=O_reg_size, r_cost=0.03, w_cost=0.03, area=0, bank=1,
                                random_bank_access=False, r_port=2, w_port=2, rw_port=0, latency=1)

        self.reg_dict = {'I': reg_IW, 'W': reg_IW, 'O': reg_O}

    def set_served_dimensions(self):
        svd_dim_I = [i for i, unroll in enumerate(self.MAC_unroll.items()) if unroll[1][0] == 'K']
        svd_dim_W = [i for i, unroll in enumerate(self.MAC_unroll.items()) if unroll[1][0] in ['B', 'OX', 'OY']]
        svd_dim_O = [i for i, unroll in enumerate(self.MAC_unroll.items()) if unroll[1][0] in ['C', 'FX', 'FY']]

        served_dimensions_I = {tuple(1 if i == num else 0 for i in range(self.unroll_counts)) for num in svd_dim_I}
        served_dimensions_W = {tuple(1 if i == num else 0 for i in range(self.unroll_counts)) for num in svd_dim_W}
        served_dimensions_O = {tuple(1 if i == num else 0 for i in range(self.unroll_counts)) for num in svd_dim_O}

        self.svd_dim = {
            'I': svd_dim_I,
            'W': svd_dim_W,     # [2,3] means W served (broadcast) dimension OX and OY if MAC_unroll is {'D1': ('K', 4), 'D2': ('C', 16), 'D3': ('OX', 2), 'D4': ('OY', 2)}
            'O': svd_dim_O
        }
        self.served_dimensions_dict = {'I': served_dimensions_I, 'W': served_dimensions_W, 'O': served_dimensions_O}
        logger.debug(f'served dimensions: {self.served_dimensions_dict}')

    def set_lb_bandwidth(self):
        """ self.dimension + self.svd_dim --> reg_counts --> lb_bandwidth

        self.dimension is like: {'D1': 8, 'D2': 32, 'D3': 2, 'D4': 2}
        """
        ''' calculate reg counts '''
        reg_counts: Dict[str, int] = {}
        unroll_dimension_list = list(self.dimensions.values())
        Total_MACs = math.prod(unroll_dimension_list)
        for op, svd_dim in self.svd_dim.items():
            reg_counts[op] = Total_MACs
            for id in svd_dim:
                reg_counts[op] = reg_counts[op] / unroll_dimension_list[id]
        ''' calculate local buffer bandwidth to match reg data transfer '''
        lb_bandwidth = {op: self.reg_dict[op].w_bw * reg_counts for op, reg_counts in reg_counts.items()}
        self.reg_counts = reg_counts
        self.Total_MACs = Total_MACs
        self.lb_bandwidth = lb_bandwidth
        logger.debug(f'reg counts: {self.reg_counts}; local buffer bandwidth: {self.lb_bandwidth}; Total MACs = {self.Total_MACs}')

    def lb_instance_dut(self):
        # 设置每一个local buffer
        lb_instance_list: List[Dict[str, MemoryInstance | str]] = []
        for local_buffer in self.local_buffers:
            op, size = local_buffer.values()  # 这个 op 可能是操作数共享的，例如 'I/O' 这样的
            name = f'sram_{size/8192}KB'  # bit -> KB
            bw = max([self.lb_bandwidth[_op] for _op in op.split('/')])  # 如果是操作数共享的，那么挑一个最大的带宽去匹配
            r_bw, w_bw = (bw, bw)
            r_cost = self._estimate_port_cost(True, size, r_bw)
            w_cost = self._estimate_port_cost(False, size, w_bw)
            single_lb = MemoryInstance(name=name, size=size, r_bw=r_bw, w_bw=w_bw, r_cost=r_cost, w_cost=w_cost,
                                       area=0, bank=1, random_bank_access=True, r_port=1, w_port=1, rw_port=0,      # 这两行的入参不可传参配置
                                       latency=1, min_r_granularity=64, min_w_granularity=64)
            lb_instance_list.append({'op': op, 'mem_instance': single_lb})

        self.lb_instance_list = lb_instance_list

    @staticmethod
    def _estimate_port_cost(is_read: bool, size: int, bw: int) -> float:
        """根据buffer 的size 和bandwidth, 估计read 或write 一次的energy

        Args:
            r_or_w (str): read or write
            size (int): ? bit
            bw (int): ? bit/cc

        Returns:
            float: read cost and write cost

        scale:
            先固定最小的单元: 8KB size and 8Bytes/cc 的 read 单次能量为 4(J)

            然后size 或bw 每翻两倍, port read energy 翻一倍

            最后在 read energy 的基础上增加 50% 得到 write energy
        """
        # 先单位换算，size 换成KB， bw 换成 B/cc
        size = size / (1024 * 8)
        bw = bw / 8
        cost_unit_at_8bw_8size = 4
        read_cost = cost_unit_at_8bw_8size * math.sqrt((size / 8) * (bw / 8))
        if is_read:
            return read_cost
        else:  # write 能量被认为大于 read能量，且比例为1.5
            return read_cost * 1.5

    def gb_instance_dut(self):
        op, size, bw = self.global_buffer['op'], self.global_buffer['size'], self.global_buffer['bandwidth']
        r_cost = self._estimate_port_cost(True, size, bw)
        w_cost = self._estimate_port_cost(False, size, bw)
        gb = MemoryInstance(name=f"sram_{size/(1024*1024*8)}MB_A", size=size, r_bw=bw, w_bw=bw, r_cost=r_cost, w_cost=w_cost,
                           area=0, bank=1, random_bank_access=True, r_port=1, w_port=1, rw_port=0,      # 这两行的入参不可传参配置
                           latency=1, min_r_granularity=64, min_w_granularity=64)
        self.gb = gb

    def dram_instance_dut(self):
        op, bw = self.dram['op'], self.dram['bandwidth']
        size = sz if (sz := self.dram.get('size')) else 10000000000
        ddr = MemoryInstance(name="dram", size=size, r_bw=bw, w_bw=bw,
                            r_cost=700, w_cost=750, area=0, bank=1, random_bank_access=False,                    # 这两行的入参不可传参配置
                            r_port=0, w_port=0, rw_port=1, latency=1)
        self.ddr = ddr

    def memory_hierarchy_dut(self) -> MemoryHierarchy:

        memory_hierarchy_graph = MemoryHierarchy(operational_array=self.multiplier_array)
        '''
        fh: from high = wr_in_by_high
        fl: from low = wr_in_by_low
        th: to high = rd_out_to_high
        tl: to low = rd_out_to_low
        '''
        # mem level 使用的operands 为 I1, I2, O，而入参使用的是 I, W, O，需要一个dict 进行转换
        link = {'I': 'I1', 'W': 'I2', 'O': 'O'}

        # ADD reg to mem. hier. graph
        self.reg_dict.pop('I')  #* firstly, delete I-reg for better support G-unrolling
        logger.debug(f'delete I-reg for better support G-unrolling, NOW, reg dict is {self.reg_dict}')
        for op, reg in self.reg_dict.items():
            memory_hierarchy_graph.add_memory(memory_instance=reg,
                                              operands=(link[op], ),
                                              port_alloc=self._set_port_alloc([op], reg),
                                              served_dimensions=self.served_dimensions_dict[op])

        # ADD local buffer to mem. hier. graph
        for lb_dict in self.lb_instance_list:  # 这里的 op 可能是共享的，如 'I/O'
            op = lb_dict['op']
            lb = lb_dict['mem_instance']
            op_li = [_op for _op in op.split('/')]
            op_li = [link[_op] for _op in op_li]  # 利用 link 进行操作数表达替换
            operands = tuple(op_li)
            memory_hierarchy_graph.add_memory(memory_instance=lb,
                                              operands=operands,
                                              port_alloc=self._set_port_alloc(op_li, lb),
                                              served_dimensions='all')

        # ADD GB or/and DDR to mem. hier. graph
        if self.global_buffer:
            op_li = [_op for _op in self.global_buffer['op'].split('/')]
            op_li = [link[_op] for _op in op_li]  # 利用 link 进行操作数表达替换
            operands = tuple(op_li)
            memory_hierarchy_graph.add_memory(memory_instance=self.gb,
                                              operands=operands,
                                              port_alloc=self._set_port_alloc(op_li, self.gb),
                                              served_dimensions='all')
        if self.dram:
            op_li = [_op for _op in self.dram['op'].split('/')]
            op_li = [link[_op] for _op in op_li]  # 利用 link 进行操作数表达替换
            operands = tuple(op_li)
            memory_hierarchy_graph.add_memory(memory_instance=self.ddr,
                                              operands=operands,
                                              port_alloc=self._set_port_alloc(op_li, self.ddr),
                                              served_dimensions='all')

        self.memory_hierarchy = memory_hierarchy_graph
        logger.debug(f'Create Memory Hierarchy sucssessfully, totally {memory_hierarchy_graph.number_of_nodes()} nodes')
        logger.debug(f'Memory Level Nodes are: {[node for node in nx.topological_sort(memory_hierarchy_graph)]}')

    def _set_port_alloc(self, op_li, mem: MemoryInstance) -> Tuple[Dict]:
        port_alloc = []
        if mem.r_port == 1 and mem.w_port == 1 and mem.rw_port == 0:  # I/W reg, local buffer and global buffer
            for _op in op_li:
                port_alo = {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'} if _op == 'O' else {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None}
                port_alloc.append(port_alo)
            return tuple(port_alloc)

        elif mem.r_port == 2 and mem.w_port == 2:  # O reg
            for _op in op_li:
                port_alo = {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'} if _op == 'O' else {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None}
                port_alloc.append(port_alo)
            return tuple(port_alloc)

        elif mem.rw_port == 1:  # dram
            for _op in op_li:
                port_alo = {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'} if _op == 'O' else {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None}
                port_alloc.append(port_alo)
            return tuple(port_alloc)

        else:
            raise BufferError(
                f'memory instance r/w/rw port set ERROR!, number of ports are: r_port: {mem.r_port}, w_port: {mem.w_port}, rw_port: {mem.rw_port}')

    def core_dut(self):
        self.core1 = Core(self.core_id, operational_array=self.multiplier_array, memory_hierarchy=self.memory_hierarchy)
        logger.info(f'Create Core Sucssessfully !!! core id = {self.core1.id}')

    @staticmethod
    def get_hardware_config(core: Core):
        multiplier_array = core.operational_array
        config = {"MAC Array": multiplier_array, "Mem hier": {}}
        memhier = core.get_memory_hierarchy()
        for id, mem_lv in enumerate(nx.topological_sort(memhier)):
            mem_lv: MemoryLevel
            mem_ins = mem_lv.memory_instance
            operands = mem_lv.operands
            served_dim = mem_lv.served_dimensions
            mem_lv_info = {"mem lv": mem_lv, "mem ins": mem_ins, "operands": operands, "served dim": served_dim}
            config['Mem hier'][id] = mem_lv_info
        return config

    def __jsonrepr__(self):
        return self.get_hardware_config(self.get_core())

    def save_to_json(self, file_path='./output_core_config.json'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as fp:
            json.dump(self, fp, default=self.complexHandler, indent=4)
        logger.info(f'export hardware config info to {file_path}')

    @staticmethod
    def complexHandler(obj):
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "__jsonrepr__"):
            return obj.__jsonrepr__()
        else:
            raise TypeError(f"Object of type {type(obj)} is not serializable. Create a __jsonrepr__ method.")


if __name__ == '__main__':
    core_info = {
        'MAC_unroll': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'local_buffers': [{
            'op': 'W',
            'size': 262144
        }, {
            'op': 'W',
            'size': 8388608
        }, {
            'op': 'I/O',
            'size': 524288
        }],
        'global_buffer': {
            'op': 'O/I',
            'size': 8388608,
            'bandwidth': 1024
        },
        'dram': {
            'op': 'W/I/O',
            'size': 0,
            'bandwidth': 64
        }
    }
    coregen = CoreGenerator.from_dict(core_info)
    print(coregen.lb_instance_list)
    print(coregen.local_buffers)