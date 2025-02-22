from math import ceil
import numpy as np
from npuperf.classes.mapping.combined_mapping import FourWayDataMoving
from npuperf.classes.hardware.architecture.core import Core
from npuperf.classes.hardware.architecture.memory_level import MemoryLevel
# from classes.workload.layer_node import LayerNode
# from classes.mapping.spatial.spatial_mapping import SpatialMapping
from npuperf.utils import pickle_deepcopy
from typing import List, Dict, Tuple


def extract_port_latency(port_busy_time: List[Tuple[int, int]]):
    """
    提取端口延迟，如 [(3, 5), (5, 9), (1, 6)]，那么返回 max(9, 1+11), 9和1分别是最大和最小的 end cycle和start cycle, 11 = 2+4+5 \\
    Given a list of port's busy time collection, this function returns the maximal latency this port can cause. \\
    E.g., port_busy_time = [(4,7), (3,6)] means that the port need to be busy in cycle (4,5,6) and cycle (3,4,5).
    """
    if len(port_busy_time) == 1:
        return port_busy_time[0][1]     # list只有一个tuple，例如 [(4, 7)], 那么返回 7
    else:
        start_cycle = min([cc[0] for cc in port_busy_time])     # tuple 第一位的最小值
        largest_cycle = max([cc[1] for cc in port_busy_time])   # tuple 第二位的最大值
        sum_indicators = np.zeros(largest_cycle, dtype=np.int8)
        for (start, end) in port_busy_time:
            indicator = np.zeros(largest_cycle, dtype=np.int8)
            indicator[start:end] = 1
            sum_indicators += indicator
        total_busy_cycle = int(sum(sum_indicators))
        return max(largest_cycle, start_cycle + total_busy_cycle)


class DataCopyAction:
    """
    DataCopyAction collects information for copying certain amount of data
    from one memory level to another memory level.
    """

    def __init__(self, data_amount: int, data_source: Tuple[str, int], data_destination: Tuple[str, int], core: Core):
        '''e.g.:    \\
        data_amount1 = 8600  # bit  \\
        data_source1 = ('O', 0)     \\
        data_destination1 = ('I1', 1)

        好像搞懂了这个source 和destination 的表示方法，应该对照画出来的 Memory Hierarchy 图看.   \\
        ('O', 1) 表示:能够存放O的内存里面, 第0级, 也就是最低的一级   \\
        ('I1', 1) 表示:能够存放 I1的内存里面, 第1级, 也就是从下往上第二个有'I1'表示的内存
        '''

        self.data_amount = data_amount
        self.source_op = data_source[0]
        self.source_lv = data_source[1]
        self.dest_op = data_destination[0]
        self.dest_lv = data_destination[1]
        self.data_source_mem: MemoryLevel = core.get_memory_hierarchy_dict()[self.source_op][self.source_lv]
        self.data_dest_mem: MemoryLevel = core.get_memory_hierarchy_dict()[self.dest_op][self.dest_lv]
        self.core = core

        self.extract_data_copy_mem_chain()
        self.calc_energy_and_latency()

    def extract_data_copy_mem_chain(self):
        """
        内存传送链, 从哪个mem, 经过哪个mem, 传到哪个mem, 还有哪些端口在其中被用到   \\
        This function extract the data send path in the memory hierarchy,
        i.e., from which mem, through which mems, to which mem,
        and which memory ports are used in between.
        """
        core = self.core
        self.source_op2 = self.source_op
        self.source_lv2 = self.source_lv
        self.dest_op2 = self.dest_op
        self.dest_lv2 = self.dest_lv
        if self.source_op != self.dest_op:
            shared_mem = core.get_lowest_shared_mem_level_above(
                self.source_op, self.source_lv, self.dest_op, self.dest_lv)

            if shared_mem == self.data_source_mem:
                self.source_op2 = self.dest_op
                self.source_lv2 = shared_mem.mem_level_of_operands[self.dest_op]

            elif shared_mem == self.data_dest_mem:
                self.dest_op2 = self.source_op
                self.dest_lv2 = shared_mem.mem_level_of_operands[self.source_op]

        if self.source_op2 != self.dest_op2:
            ''' When the source operand and the destination operand are not the same,
            the data must firstly go up to a shared memory level between that 2 operands
            and then go down to the destination memory level '''

            ''' data_copy_mem_chain initialization '''
            out_port = [port for port in self.data_source_mem.port_list
                        if (self.source_op, self.source_lv, 'rd_out_to_high') in port.served_op_lv_dir][0]
            data_copy_mem_chain = [
                [self.data_source_mem,
                 {'in': None,
                  'out': (self.source_op, self.source_lv, 'rd_out_to_high'),
                  'in_port': None,
                  'out_port': out_port}]
            ]

            # shared_mem = core.get_lowest_shared_mem_level_above(
            #     self.source_op, self.source_lv, self.dest_op, self.dest_lv)

            source_lv_end = shared_mem.mem_level_of_operands[self.source_op]
            dest_lv_start = shared_mem.mem_level_of_operands[self.dest_op]

            ''' firstly go up '''
            for i in range(self.source_lv + 1, source_lv_end, +1):
                mem = core.get_memory_hierarchy_dict()[self.source_op][i]
                in_port = [port for port in mem.port_list
                           if (self.source_op, i, 'wr_in_by_low') in port.served_op_lv_dir][0]
                out_port = [port for port in mem.port_list
                            if (self.source_op, i, 'rd_out_to_high') in port.served_op_lv_dir][0]
                data_copy_mem_chain.append(
                    [mem,
                     {'in': (self.source_op, i, 'wr_in_by_low'),
                      'out': (self.source_op, i, 'rd_out_to_high'),
                      'in_port': in_port,
                      'out_port': out_port}]
                )

            ''' at the top '''
            mem = core.get_memory_hierarchy_dict()[self.source_op][source_lv_end]
            in_port = [port for port in mem.port_list
                       if (self.source_op, source_lv_end, 'wr_in_by_low') in port.served_op_lv_dir][0]
            out_port = [port for port in mem.port_list
                        if (self.dest_op, dest_lv_start, 'rd_out_to_low') in port.served_op_lv_dir][0]
            data_copy_mem_chain.append(
                [mem,
                 {'in': (self.source_op, source_lv_end, 'wr_in_by_low'),
                  'out': (self.dest_op, source_lv_end, 'rd_out_to_low'),
                  'in_port': in_port,
                  'out_port': out_port}]
            )

            ''' then go down '''
            for i in range(dest_lv_start - 1, self.dest_lv, -1):
                mem = core.get_memory_hierarchy_dict()[self.dest_op][i]
                in_port = [port for port in mem.port_list
                           if (self.dest_op, i, 'wr_in_by_high') in port.served_op_lv_dir][0]
                out_port = [port for port in mem.port_list
                            if (self.dest_op, i, 'rd_out_to_low') in port.served_op_lv_dir][0]
                data_copy_mem_chain.append(
                    [mem,
                     {'in': (self.dest_op, i, 'wr_in_by_high'),
                      'out': (self.dest_op, i, 'rd_out_to_low'),
                      'in_port': in_port,
                      'out_port': out_port}]
                )

            ''' reach the destination mem '''
            in_port = [port for port in self.data_dest_mem.port_list
                       if (self.dest_op, self.dest_lv, 'wr_in_by_high') in port.served_op_lv_dir][0]
            data_copy_mem_chain.append(
                [self.data_dest_mem,
                 {'in': (self.dest_op, self.dest_lv, 'wr_in_by_high'),
                  'out': None,
                  'in_port': in_port,
                  'out_port': None}]
            )

        else:
            ''' When the source operand and the destination operand are the same,
            the data copy path is one-directional, either go up or go down '''

            if self.source_lv2 < self.dest_lv2:
                ''' Go up. data_copy_mem_chain initialization '''
                out_port = [port for port in self.data_source_mem.port_list
                            if (self.source_op2, self.source_lv2, 'rd_out_to_high') in port.served_op_lv_dir][0]
                data_copy_mem_chain = [
                    [self.data_source_mem,
                     {'in': None,
                      'out': (self.source_op2, self.source_lv2, 'rd_out_to_high'),
                      'in_port': None,
                      'out_port': out_port}]
                ]

                for i in range(self.source_lv2 + 1, self.dest_lv2, +1):
                    mem = core.get_memory_hierarchy_dict()[self.source_op2][i]
                    in_port = [port for port in mem.port_list
                               if (self.source_op2, i, 'wr_in_by_low') in port.served_op_lv_dir][0]
                    out_port = [port for port in mem.port_list
                                if (self.source_op2, i, 'rd_out_to_high') in port.served_op_lv_dir][0]
                    data_copy_mem_chain.append(
                        [mem,
                         {'in': (self.source_op2, i, 'wr_in_by_low'),
                          'out': (self.source_op2, i, 'rd_out_to_high'),
                          'in_port': in_port,
                          'out_port': out_port}]
                    )

                ''' reach the destination mem '''
                in_port = [port for port in self.data_dest_mem.port_list
                           if (self.dest_op2, self.dest_lv2, 'wr_in_by_low') in port.served_op_lv_dir][0]
                data_copy_mem_chain.append(
                    [self.data_dest_mem,
                     {'in': (self.dest_op2, self.dest_lv2, 'wr_in_by_low'),
                      'out': None,
                      'in_port': in_port,
                      'out_port': None}]
                )

            else:
                ''' Go down. data_copy_mem_chain initialization '''
                out_port = [port for port in self.data_source_mem.port_list
                            if (self.source_op2, self.source_lv2, 'rd_out_to_low') in port.served_op_lv_dir][0]
                data_copy_mem_chain = [
                    [self.data_source_mem,
                     {'in': None,
                      'out': (self.source_op2, self.source_lv2, 'rd_out_to_low'),
                      'in_port': None,
                      'out_port': out_port}]
                ]

                for i in range(self.source_lv2 - 1, self.dest_lv2, -1):
                    mem = core.get_memory_hierarchy_dict()[self.source_op2][i]

                    in_port = [port for port in mem.port_list if (self.dest_op2, i, 'wr_in_by_high') in port.served_op_lv_dir][0]
                    out_port = [port for port in mem.port_list if (self.dest_op2, i, 'rd_out_to_low') in port.served_op_lv_dir][0]
                    data_copy_mem_chain.append(
                        [mem,
                         {'in': (self.dest_op2, i, 'wr_in_by_high'),
                          'out': (self.dest_op2, i, 'rd_out_to_low'),
                          'in_port': in_port,
                          'out_port': out_port}]
                    )

                ''' reach the destination mem '''
                in_port = [port for port in self.data_dest_mem.port_list
                           if (self.dest_op2, self.dest_lv2, 'wr_in_by_high') in port.served_op_lv_dir][0]
                data_copy_mem_chain.append(
                    [self.data_dest_mem,
                     {'in': (self.dest_op2, self.dest_lv2, 'wr_in_by_high'),
                      'out': None,
                      'in_port': in_port,
                      'out_port': None}]
                )

        self.data_copy_mem_chain = data_copy_mem_chain

    def calc_energy_and_latency(self):
        """ This function calculates the total energy, energy breakdown, latency breakdown,
        and ports' busy cycle range of each memory's each port in the data_copy_mem_chain
        for this data copying action """

        # 这 5 个是self的
        # 关于energy breakdown的解释：如果说从A 传到B，A和B之间隔了两个mem lv，那么data_copy_mem_chain 就有四个mem lv，而之间有三级的传递，energy breakdown就有六个数值
        # 这是因为每一级传递都分为 send energy 和 recive energy两部分
        energy = 0
        energy_breakdown = []
        energy_breakdown_further = {}
        latency_breakdown = []
        port_active_cycle = {}

        timeline_cc = 0

        # initialize energy_breakdown_further (to unify the energy breakdown format with normal layer)
        for operand, mem_list in self.core.mem_hierarchy_dict.items():
            ''' For every memory, there are 4 data transfer link in the hierarchy:
            rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high '''
            energy_breakdown_further[operand] = [FourWayDataMoving(0, 0, 0, 0) for i in range(len(mem_list))]

        for idx, (send_mem, send_port) in enumerate(self.data_copy_mem_chain):
            if send_mem == self.data_dest_mem:
                self.energy = energy
                self.energy_breakdown = energy_breakdown
                self.energy_breakdown_further = energy_breakdown_further
                self.latency_breakdown = latency_breakdown
                self.port_active_cycle = port_active_cycle
                break
            else:
                receive_mem, receive_port = self.data_copy_mem_chain[idx + 1]

                ''' energy '''
                send_energy = ceil(self.data_amount / send_port['out_port'].port_bw_min) * \
                              (send_port['out_port'].port_bw_min / send_port['out_port'].port_bw) * \
                              send_mem.read_energy
                mem_op = send_port['out'][0]
                mem_lv = send_port['out'][1]
                mov_dir = send_port['out'][2]
                energy_breakdown_further[mem_op][mem_lv].update_single_dir_data(mov_dir, send_energy)
                energy += send_energy

                receive_energy = ceil(self.data_amount / receive_port['in_port'].port_bw_min) * \
                                 (receive_port['in_port'].port_bw_min / receive_port['in_port'].port_bw) * \
                                 receive_mem.write_energy
                mem_op = receive_port['in'][0]
                mem_lv = receive_port['in'][1]
                mov_dir = receive_port['in'][2]
                energy_breakdown_further[mem_op][mem_lv].update_single_dir_data(mov_dir, receive_energy)
                energy += receive_energy

                energy_breakdown.extend([send_energy, receive_energy])

                ''' latency '''
                send_cc = ceil(self.data_amount / send_port['out_port'].port_bw)
                receive_cc = ceil(self.data_amount / receive_port['in_port'].port_bw)

                actual_cc = max(send_cc, receive_cc)
                latency_breakdown.append(actual_cc)

                '''
                Generate a dict that covers all the port's busy cycle range.
                e.g., if data is transferred from mem0 to mem2, in which
                      portA of mem0 need to be busy from cycle 0 to cycle 80,
                      portB of mem1 need to be busy from cycle 81 to cycle 90,
                it will generate: {portA_id: (0,80), portB_id: 81,90)}
                '''
                send_port_id = send_port['out_port'].port_id
                receive_port_id = receive_port['in_port'].port_id
                if send_port_id in port_active_cycle.keys():
                    port_active_cycle[send_port_id].append((timeline_cc, timeline_cc + actual_cc))
                else:
                    port_active_cycle[send_port_id] = [(timeline_cc, timeline_cc + actual_cc)]
                if receive_port_id in port_active_cycle.keys():
                    port_active_cycle[receive_port_id].append((timeline_cc, timeline_cc + actual_cc))
                else:
                    port_active_cycle[receive_port_id] = [(timeline_cc, timeline_cc + actual_cc)]

                    timeline_cc += actual_cc

    def __repr__(self):
        return f"{self.data_amount} bit, from {self.source_op, self.source_lv}, " \
               f"to {self.dest_op, self.dest_lv}"

    def __str__(self):
        return f"{self.data_amount} bit, from {self.source_op, self.source_lv}, " \
               f"to {self.dest_op, self.dest_lv}. \n Mem chain: {self.data_copy_mem_chain}"


class DataCopyLayer:
    """
    DataCopyLayer collects all the DataCopyActions that can happen in parallel and
    calculate their total energy and latency.
    When calculate latency, we take into account the concurrent data transferring
    possibility from multi-port memory levels.
    """

    def __init__(self, layer_id: int, data_copy_actions: List[DataCopyAction], accelerator, core_id: int):
        self.id = layer_id
        self.data_copy_actions = data_copy_actions
        self.accelerator = accelerator              # 加速器
        self.core = accelerator.get_core(core_id)   # core
        self.core_allocation = core_id              # core_id
        self.layer = self       # ??????这么写？？？
        self.MAC_energy = 0
        self.run()
        # for action in self.data_copy_actions:
        #     print(f'action={str(action)}')

    def run(self):
        for action in self.data_copy_actions:
            action.calc_energy_and_latency()        # 对每个 data_copy_action, 评估能量和延迟

        self.combine_energy()       # 汇总功耗
        self.combine_latency()      # 组合延迟，同时考虑并发

    def combine_energy(self):
        """
        Combine energy and from each DataCopyAction by summing them up.
        """
        self.energy_total = sum([action.energy for action in self.data_copy_actions])

        energy_breakdown_further = {}
        for operand, mem_list in self.core.mem_hierarchy_dict.items():
            energy_breakdown_further[operand] = [FourWayDataMoving(0, 0, 0, 0) for i in range(len(mem_list))]

        for action in self.data_copy_actions:
            for operand, mem_list in self.core.mem_hierarchy_dict.items():
                for lv in range(len(mem_list)):
                    energy_breakdown_further[operand][lv] += action.energy_breakdown_further[operand][lv]
        self.energy_breakdown_further = energy_breakdown_further

    def combine_latency(self):
        """
        Combine latency from each DataCopyAction taking into account the memory port concurrency.
        """

        port_active_cycle_collect = {}
        for action in self.data_copy_actions:
            for port_id, active_range in action.port_active_cycle.items():
                if port_id in port_active_cycle_collect.keys():
                    port_active_cycle_collect[port_id].extend(active_range)
                else:
                    port_active_cycle_collect[port_id] = active_range

        port_latency_collect = {}
        for port_id, port_busy_time in port_active_cycle_collect.items():
            port_latency_collect[port_id] = extract_port_latency(port_busy_time)

        self.port_active_cycle_collect = port_active_cycle_collect
        self.port_latency_collect = port_latency_collect
        self.latency_total = max([va for va in port_latency_collect.values()]) if port_latency_collect else 0
        self.latency_total1 = self.latency_total
        self.latency_total2 = self.latency_total

    def __add__(self, other):
        sum = pickle_deepcopy(self)
        sum.energy_total += other.energy_total
        for op in sum.energy_breakdown_further.keys():
            l = []
            for i in range(min(len(self.energy_breakdown_further[op]), len(other.energy_breakdown_further[op]))):
                l.append(self.energy_breakdown_further[op][i] + other.energy_breakdown_further[op][i])
            i = min(len(self.energy_breakdown_further[op]), len(other.energy_breakdown_further[op]))
            l += self.energy_breakdown_further[op][i:]
            l += other.energy_breakdown_further[op][i:]
            sum.energy_breakdown_further[op] = l
        sum.latency_total += other.latency_total
        sum.latency_total1 += other.latency_total1
        sum.latency_total2 += other.latency_total2
        return sum

    def __mul__(self, number):
        mul = pickle_deepcopy(self)
        mul.energy_breakdown_further = {
            op: [
                mul.energy_breakdown_further[op][i] * number for i in range(len(mul.energy_breakdown_further[op]))
            ] for op in mul.energy_breakdown_further.keys()
        }
        mul.energy_total *= number
        mul.latency_total *= number
        mul.latency_total1 *= number

        return mul

    def __str__(self):
        return f'DataCopyLayer_{self.id}'
    def __repr__(self):
        return str(self)

if __name__ == "__main__":
    # from inputs.HW.ISP_NPUv2 import accelerator
    from inputs.HW.Meta_prototype_DF import accelerator

    layer_id = 1
    core_id = 1
    core = accelerator.get_core(core_id)

    data_amount1 = 8600  # bit
    data_source1 = ('O', 0)
    data_destination1 = ('I1', 1)

    data_amount2 = 11200  # bit
    data_source2 = ('I1', 2)
    data_destination2 = ('I1', 1)

    data_amount3 = 7000  # bit
    data_source3 = ('O', 1)
    data_destination3 = ('O', 2)

    data_copy_action1 = DataCopyAction(data_amount1, data_source1, data_destination1, core)
    data_copy_action2 = DataCopyAction(data_amount2, data_source2, data_destination2, core)
    data_copy_action3 = DataCopyAction(data_amount3, data_source3, data_destination3, core)

    data_copy_actions = [data_copy_action1, data_copy_action2, data_copy_action3]

    dcl = DataCopyLayer(layer_id, data_copy_actions, accelerator, core_id)
    a = 1

    print(data_copy_action1.data_copy_mem_chain)
    print(data_copy_action1.data_dest_mem)
    print(data_copy_action1.data_source_mem)
    print(data_copy_action1.energy)
    print(data_copy_action1.energy_breakdown)
    print()
    print(dcl.layer)
    print(dcl.port_active_cycle_collect)
    print(dcl.port_latency_collect)
    print(dcl.MAC_energy)

# data_copy_mem_chain
# [
#     [
#         MemoryLevel(instance=rf_2B,operands=['O'],served_dimensions={Dimension(id=1,name=D2,size=2)}),
#         {'in': None, 'out': ('O', 0, 'rd_out_to_high'), 'in_port': None, 'out_port': r_port_2}
#     ],

#     [
#         MemoryLevel(instance=sram_64KB,operands=['I1', 'O'],served_dimensions={Dimension(id=0,name=D1,size=32), Dimension(id=1,name=D2,size=2), Dimension(id=3,name=D4,size=4), Dimension(id=2,name=D3,size=4)}),
#         {'in': ('O', 1, 'wr_in_by_low'), 'out': ('O', 1, 'rd_out_to_high'), 'in_port': w_port_1, 'out_port': r_port_1}
#     ],

#     [
#         MemoryLevel(instance=sram_1MB_A,operands=['I1', 'O'],served_dimensions={Dimension(id=0,name=D1,size=32), Dimension(id=1,name=D2,size=2), Dimension(id=3,name=D4,size=4), Dimension(id=2,name=D3,size=4)}),
#         {'in': ('O', 2, 'wr_in_by_low'), 'out': None, 'in_port': w_port_1, 'out_port': None}
#     ]
# ]