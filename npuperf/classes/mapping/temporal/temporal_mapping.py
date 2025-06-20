from typing import Dict
from copy import deepcopy
from math import prod
from npuperf.classes.workload.layer_node import LayerNode
from npuperf.utils import pickle_deepcopy

class TemporalMapping:
    """
    Class that collect all the info related to temporal mapping.
    """

    def __init__(self, temporal_mapping_dict: Dict, layer_node: LayerNode):
        self.mapping_dic_origin = temporal_mapping_dict
        self.layer_node = layer_node
        self.operand_list = layer_node.operand_list

        ''' Extract memory hierarchy level count for each operand from temporal mapping definition '''
        self.mem_level = {op: len(tmap) for (op, tmap) in temporal_mapping_dict.items()}

        ''' For each memory level, if the innermost/bottom loop is ir loop, merge it down to the below level '''
        ''' 不会有一个mem lvl, 其最底侧是 ir 的, 因为要是它是 ir 那肯定可以合并到下一个 mem lv里 '''
        self.innermost_stationary_loop_merge_down()

        ''' Calculate the current and below level (cabl) iteration cycle for each memory level,
        i.e., each memory level refreshes once, how many cycles it covers '''
        self.calc_cycle_cabl_level()

        # ''' Calculate the current and below loop (cabl) iteration cycle for each loop,
        # i.e., each loop iterates once, how many cycles it covers '''
        # self.calc_cycle_cabl_loop()

        ''' Calculate the top-ir loop size at each memory level, which will be used
        to compute instant required memory BW in combined_mapping.py '''
        ''' 计算每个mem lv 的最高一级的 ir size (在之前我们已经将上一级最后的 ir 合并到本级), 这个用于计算 即时的内存所需带宽 '''
        self.calc_top_r_and_ir_loop()

    def __str__(self):
        return f"TemporalMapping({self.mapping_dic_origin})"

    def __repr__(self):
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation of this object to save it to a json file.
        """
        return {"temporal_mapping": self.mapping_dic_origin}

    def innermost_stationary_loop_merge_down(self):
        """
        Iteratively merging down the ir loops which located at the bottom position of each memory level.
        Also calculate the MAC level data stationary cycle, i,e., the innermost memory level's bottom ir loops.
        同时会计算MAC层的数据静止周期, 即最内层的mem lv 的最底层的 ir loop.
        """
        ''' Initialization '''
        mapping_current = pickle_deepcopy(self.mapping_dic_origin)
        mapping_previous = pickle_deepcopy(self.mapping_dic_origin)
        done = False

        while not done:
            mapping_st = {op: [[] for _ in range(self.mem_level[op])] for op in self.operand_list}
            MAC_level_st = {op: 1 for op in self.operand_list}
            # 遍历每个操作数
            for operand in self.mem_level.keys():
                # 遍历操作数的每个时间映射层级
                for level, current_level_loops in enumerate(mapping_previous[operand]):
                    if not current_level_loops:
                        mapping_st[operand][level] = pickle_deepcopy(current_level_loops)
                    else:
                        # 遍历里面的(OY,2) (OX,3)等
                        for loop_type, loop_dim in current_level_loops:
                            # 如果是ir的tempral loop, 则将其合并到上一级
                            if loop_type in self.layer_node.operand_loop_dim[operand]['ir']:
                                if level == 0:
                                    MAC_level_st[operand] *= loop_dim
                                    mapping_st[operand][level].append((loop_type, loop_dim))
                                    mapping_current[operand][level].remove((loop_type, loop_dim))
                                else:
                                    mapping_st[operand][level - 1].append((loop_type, loop_dim))
                                    mapping_current[operand][level].remove((loop_type, loop_dim))
                            else:
                                mapping_st[operand][level].extend(mapping_current[operand][level])
                                break
            if mapping_st != mapping_previous:
                mapping_previous = pickle_deepcopy(mapping_st)
                mapping_current = pickle_deepcopy(mapping_st)
                continue
            else:
                done = True

        self.mapping_dic_stationary = mapping_st
        self.MAC_level_data_stationary_cycle = MAC_level_st

    def calc_cycle_cabl_level(self):
        """
        Calculate the iteration cycles that each memory level covers
        """
        ''' iteration_each_level only counts for the current level for-loops '''
        iteration_each_level = {op: [
            prod([loop_dim for (_, loop_dim) in self.mapping_dic_stationary[op][lv]]) for lv in range(self.mem_level[op])
        ] for op in self.operand_list}
        ''' cycle_per_level count for current and below levels' for-loops '''
        cycle_cabl_level = {op: [
            prod(iteration_each_level[op][0:lv+1]) for lv in range(self.mem_level[op])
        ] for op in self.operand_list}

        ''' ASSERT: The total cycle count must be the same for all operand '''
        total_cycle = [cycle_cabl_level[op][-1] for op in self.operand_list]
        assert all(x == total_cycle[0] for x in total_cycle), \
            f"The total cycle count is not the same for all operand {total_cycle}, please correct the temporal mapping."

        self.cycle_cabl_level = cycle_cabl_level
        self.total_cycle = total_cycle[0]

    def calc_top_r_and_ir_loop(self):
        """
        top_ir_loop_size: For each memory level, from top to bottom, the product of top few irrelevant loops.
        top_ir is used for later required instant memory bandwidth calculation.
        """
        ''' Initialization '''
        ''' self.mem_level[op] + 1 to add the placeholder for operational array level '''
        top_r_loop_size = {op: [
            1 for _ in range(self.mem_level[op] + 1)
        ] for op in self.operand_list}

        top_ir_loop_size = {op: [
            1 for _ in range(self.mem_level[op] + 1)
        ] for op in self.operand_list}

        ''' Check and extract the top ir loops '''
        for operand in self.operand_list:
            for level, current_level_loops in enumerate(self.mapping_dic_stationary[operand]):
                if not current_level_loops:
                    continue
                else:
                    for loop_type, loop_dim in reversed(current_level_loops):
                        if loop_type in self.layer_node.operand_loop_dim[operand]['r']:
                            top_r_loop_size[operand][level + 1] *= loop_dim
                        else:
                            break
                    for loop_type, loop_dim in reversed(current_level_loops):
                        if loop_type in self.layer_node.operand_loop_dim[operand]['ir']:
                            top_ir_loop_size[operand][level + 1] *= loop_dim
                        else:
                            break

        self.top_r_loop_size = top_r_loop_size
        self.top_ir_loop_size = top_ir_loop_size


if __name__ == "__main__":
    from npuperf.inputs.WL_fromjson.Meta_prototype.workload_mv1 import workload

    layer = workload[0]
    layer_node = LayerNode(1, layer)

    # # mapping case 1 for debug
    # temporal_mapping_dic = {'W': [[], [('FX', 3), ('OX', 3)], [('OX', 7), ('FX', 3)], [('OX', 2)]],
    #                         'I': [[('FX', 3)], [('OX', 3)], [('OX', 7), ('FX', 3)], [('OX', 2)]],
    #                         'O': [[('FX', 3)], [('OX', 3)], [('OX', 7), ('FX', 3), ('OX', 2)], []]}

    # mapping case 2 for debug
    # temporal_mapping_dic = \
    #     {'W': [[('K', 4)], [('FX', 3), ('OX', 3), ('C', 2)], [('OX', 7), ('FX', 3), ('OY', 3)], [('OX', 2), ('B', 2)], [('OX', 4)]],
    #      'I': [[('K', 4), ('FX', 3)], [('OX', 3), ('C', 2)], [('OX', 7), ('FX', 3), ('OY', 3)], [('OX', 2), ('B', 2), ('OX', 4)]],
    #      'O': [[('K', 4), ('FX', 3)], [('OX', 3)], [('C', 2), ('OX', 7), ('FX', 3), ('OY', 3), ('OX', 2)], [('B', 2), ('OX', 4)]]}

    temporal_mapping_dic = {
        "O": [[("C", 4)], [("OX", 32), ("K", 8), ("FY", 3), ("FX", 3)], [("OY", 16), ("OY", 16), ("OX", 16)]],
        "W": [[], [("C", 4), ("OX", 32), ("K", 8), ("FY", 3)], [("FX", 3), ("OY", 16), ("OY", 16), ("OX", 16)]],
        "I": [[("C", 4), ("OX", 32), ("K", 8), ("FY", 3), ("FX", 3)], [("OY", 16), ("OY", 16), ("OX", 16)]]
    }
    temporal_mapping = TemporalMapping(temporal_mapping_dic, layer_node)
    print(temporal_mapping.MAC_level_data_stationary_cycle) # {'O': 4, 'W': 1, 'I': 1}
    print(temporal_mapping.cycle_cabl_level)    # {'O': [4, 9216, 37748736], 'W': [1, 3072, 37748736], 'I': [9216, 37748736]}
    print(temporal_mapping.total_cycle) # 37748736
    print(temporal_mapping.top_ir_loop_size)    # {'O': [1, 4, 9, 1], 'W': [1, 1, 1, 4096], 'I': [1, 1, 1]}
    print(temporal_mapping.top_r_loop_size) # {'O': [1, 1, 1, 4096], 'W': [1, 1, 24, 1], 'I': [1, 1, 1]}
