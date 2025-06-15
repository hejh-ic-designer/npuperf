from typing import Dict
from math import prod, ceil, floor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from npuperf.classes.workload.layer_node import LayerNode
import npuperf.classes.mapping.mapping_assist_funcs as mapping_assist_funcs


class SpatialMapping:
    """
    Class that collect all the info related to spatial mapping.
    """

    def __init__(self, spatial_mapping_dict: Dict, layer_node: 'LayerNode'):
        self.mapping_dict_origin = spatial_mapping_dict
        # pr 解耦，将OX FX和OY FY中的OX OY解耦成r部分和ir部分
        self.mapping_dict_reform = mapping_assist_funcs.decouple_pr_loop(spatial_mapping_dict, layer_node)
        self.layer_node = layer_node
        self.operand_list = layer_node.operand_list

        ''' Extract architecture level count for each operand from spatial mapping definition, starting from MAC level '''
        self.arch_level = {op: len(smap) for (op, smap) in spatial_mapping_dict.items()}

        ''' Calculate unrolled loop size for different loop types (r/ir/total) '''
        self.calc_unroll_size()

        ''' Calculate total/unique/duplicate unit count '''
        self.calc_unit_count()

        ''' Calculate data serve scope: each data element serves/(is served by) how many unit at below level
        NOTE: data_serve_scope doesn't include MAC level, thus is one level less than other spatial mapping attributes. '''
        self.calc_data_serve_scope()

        ''' Calculate memory bandwidth incremental factor between architectural levels
        NOTE: mem_bw_boost_factor doesn't include MAC level, thus is one level less than other spatial mapping attributes. '''
        self.calc_mem_bw_boost_factor()

        ''' Added for loma: Get list of the spatially unrolled loops, without any information about arch levels'''
        self.save_spatial_loop_dim_size()

    def __str__(self):
        return f"SpatialMapping({self.mapping_dict_origin})"

    def __repr__(self):
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation of this object to save it to a file.
        """
        return {"spatial_mapping": self.mapping_dict_origin}

    def get_unrolling(self, op: str, level: int):
        """
        Return the unrolled loops for operand 'op' at level 'level'.
        'level' = 0 would signify the operational level.
        """
        return self.mapping_dict_origin[op][level]

    def calc_unroll_size(self):
        """
        Calculate unrolled loop size for different loop types (r/ir/total) per operand per architecture level
        """
        ''' Initialization '''
        unroll_size_r = {op: [1] * arch_lv for (op, arch_lv) in self.arch_level.items()}
        unroll_size_ir = {op: [1] * arch_lv for (op, arch_lv) in self.arch_level.items()}
        unroll_size_total = {op: [1] * arch_lv for (op, arch_lv) in self.arch_level.items()}

        ''' Go through the reformed spatial mapping and extract the unroll size '''
        for operand in self.operand_list:
            for level, current_level_loops in enumerate(self.mapping_dict_reform[operand]):
                for loop_type, loop_dim in current_level_loops:
                    if loop_type in self.layer_node.operand_loop_dim_reform[operand]['r']:
                        unroll_size_r[operand][level] *= loop_dim
                    else:
                        unroll_size_ir[operand][level] *= loop_dim
                    unroll_size_total[operand][level] *= loop_dim

        self.unroll_size_r = unroll_size_r
        self.unroll_size_ir = unroll_size_ir
        self.unroll_size_total = unroll_size_total

    def calc_unit_count(self):
        """
        Calculate total/unique/duplicate unit count per operand per architecture level
        """
        ''' Number of unit at each level (for each operand) '''
        # Added round call as number doesn't remain integer due to self.mapping_dict_reform number instability
        unit_count1 = {op: [
            round(prod(self.unroll_size_total[op][lv:self.arch_level[op]])) for lv in range(self.arch_level[op])
        ] for op in self.operand_list}

        unit_count2 = {op: [
            ceil(prod(self.unroll_size_total[op][lv:self.arch_level[op]])) for lv in range(self.arch_level[op])
        ] for op in self.operand_list}

        unit_count3 = {op: [
            floor(prod(self.unroll_size_total[op][lv:self.arch_level[op]])) for lv in range(self.arch_level[op])
        ] for op in self.operand_list}

        ''' ASSERT: The bottom level (MAC level) unit count must be the same for all operand '''
        bottom_unit_count1 = [unit_count1[op][0] for op in unit_count1.keys()]
        bottom_unit_count2 = [unit_count2[op][0] for op in unit_count2.keys()]
        bottom_unit_count3 = [unit_count3[op][0] for op in unit_count3.keys()]
        if all(x == bottom_unit_count1[0] for x in bottom_unit_count1):
            unit_count = unit_count1
        elif all(x == bottom_unit_count2[0] for x in bottom_unit_count2):
            unit_count = unit_count2
        elif all(x == bottom_unit_count3[0] for x in bottom_unit_count3):
            unit_count = unit_count3
        else:
            raise Exception(f"The MAC level unit count is not the same for all operands in any of the below format - "
                            f"round: {bottom_unit_count1}, ceil: {bottom_unit_count1}, floor: {bottom_unit_count1}, "
                            f"please correct the spatial mapping.")

        ''' Number of unit at each level that hold unique data (for each operand) '''
        unit_unique = {op: [
            prod(self.unroll_size_r[op][lv:self.arch_level[op]]) for lv in range(self.arch_level[op])
        ] for op in self.operand_list}

        ''' Number of unit at each level that hold the same data (for each operand) '''
        unit_duplicate = {op: [
            prod(self.unroll_size_ir[op][lv:self.arch_level[op]]) for lv in range(self.arch_level[op])
        ] for op in self.operand_list}

        self.unit_count = unit_count
        self.unit_unique = unit_unique
        self.unit_duplicate = unit_duplicate

    def calc_data_serve_scope(self):
        """
        Calculate data serve scope, i.e.,
        for input operands, it means that each data element is broadcast to how many unit at below level;
        for output operand, it means that how many unit add/collect their output values to one result, and push it to above level '''

        NOTE: data_serve_scope doesn't include MAC level, thus is one level less than other spatial mapping attributes.
        """
        ''' data_serve_scope is calculated by dividing unit_duplicate at current level by unit_count at one level above. '''
        # data_server_scope是通过将当前级别的unit_duplicate 除以上一级别的unit_count 来计算的
        data_serve_scope = {op: [
            self.unit_duplicate[op][lv]/self.unit_duplicate[op][lv+1] for lv in range(self.arch_level[op]-1)
        ] for op in self.operand_list}

        self.data_serve_scope = data_serve_scope

    def calc_mem_bw_boost_factor(self):
        """
        Calculate memory bandwidth incremental factor between architectural levels.

        NOTE: mem_bw_boost doesn't include MAC level, thus is one level less than other spatial mapping attributes.
        """
        ''' mem_bw_boost can calculated by either dividing unit_unique at current level by unit_count at one level above. '''
        mem_bw_boost = {op: [
            int(self.unit_unique[op][lv]/self.unit_unique[op][lv+1]) for lv in range(self.arch_level[op]-1)
        ] for op in self.operand_list}

        self.mem_bw_boost = mem_bw_boost

    def save_spatial_loop_dim_size(self):
        """
        Save the loops that were unrolled spatially in a list without any arch level information for easy access in loma.
        """
        # We take one of the input operands and go through the spatial mapping dict for that operand.
        # Which operand shouldn't matter as all operands store the same loops, but possibly at different arch levels.
        op = self.layer_node.input_operands[0]
        self.spatial_loop_dim_size = [loop for spatial_loops in self.mapping_dict_origin[op] for loop in spatial_loops]


if __name__ == "__main__":
    from npuperf.classes.workload.layer_node import LayerNode
    layer = {
        'operator_type': 'Conv',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 64, 'C': 64, 'OY': 512, 'OX': 512, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},        # partial sum 16 bit
        'operand_source': {'W': [], 'I': [-1]},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    layer_node = LayerNode(0, layer)

    # # mapping case 1 for debug
    # spatial_mapping_dic = {'W': [[], [('C', 10), ('K', 8)], [], []],
    #                        'I': [[], [('C', 10), ('K', 8)], [], []],
    #                        'O': [[], [('C', 10), ('K', 8)], []]}

    # # mapping case 2 for debug
    # spatial_mapping_dic = {'W': [[('B', 14)], [('C', 2), ('K', 8)], [], []],
    #                        'I': [[('K', 8)], [('B', 14)], [('C', 2)], []],
    #                        'O': [[('C', 2)], [('B', 14), ('K', 8)], []]}

    # mapping case 3 for debug
    spatial_mapping_dic = {'W': [[('OX', 14), ('OY', 7)], [('FX', 3)], [], []],
                           'I': [[], [('OX', 14), ('FX', 3)], [('OY', 7)], []],
                           'O': [[('FX', 3)], [('OX', 14), ('OY', 7)], []]}
    
    # spatial_mapping_dic = {
    #     'O': [[('C', 2.0)], [('K', 32.0), ('OY', 4.0), ('OX', 4.0)], [], [], []],
    #     'W': [[('OY', 4.0), ('OX', 4.0)], [('K', 32.0), ('C', 2.0)], [], [], []],
    #     'I': [[('K', 32.0), ('OY', 4.0), ('OX', 4.0), ('C', 2.0)], [], [], []]
    # }
    spatial_mapping = SpatialMapping(spatial_mapping_dic, layer_node)
    print(spatial_mapping.mapping_dict_reform)
    # print result：
    # {
    #     'O': [[('C', 2.0)], [('K', 32.0), ('OY', 4.0), ('OX', 4.0)], [], [], []], 
    #     'W': [[('OY', 4.0), ('OX', 4.0)], [('K', 32.0), ('C', 2.0)], [], [], []], 
    #     'I': [[('K', 32.0), ('IY_r', 4.0), ('IY_ir', 1.0), ('IX_r', 4.0), ('IX_ir', 1.0), ('C', 2.0)], [], [], []]    
    #  }
    ## 这里 I 没有设置 I-reg，所以全部都在 MAC level；
    ## 同时注意到出现了 IX 和 IY，并且 IX 和 IY 分为相关和不相关，而且这个例子里面IX 和 IY都是相关的
    ## 那种像 Eyeriss 那种，IX 和 IY 既有相关和不相关的，可以从这里改？