from typing import Dict, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from npuperf.classes.workload.layer_node import LayerNode
from math import prod
from copy import deepcopy
from npuperf.utils import pickle_deepcopy


class Loop:
    """
    Collect information of each single loop tuple in mapping.
    Applied range: from the lowest architectural level to the current level.
    """

    def __init__(self, loop: tuple, MAC_op: int, data_elem: int):
        self.loop = loop
        self.MAC_op = MAC_op
        self.data_elem = data_elem
        self.reuse = MAC_op / data_elem

    def __str__(self):
        return str(self.loop)

    def __repr__(self):
        return str(self.loop)


def decouple_pr_loop(mapping_dict: Dict, layer_node: 'LayerNode'):
    """
    This function decouples the pr loops into data size (r loops) and data reuse (ir loops).
    It also provides a transferred mapping dictionary in which the pr loops are replaced by r and ir loops.
    """
    # 其实这里都是固定的，因为只有I有pr型依赖关系
    operand_loop_dim = layer_node.operand_loop_dim
    r_ir_operand_loop_LUT = {op: relevance['r'] + relevance['ir'] for (op, relevance) in operand_loop_dim.items()}
    pr_operand_loop_LUT = {op: relevance['pr'] for (op, relevance) in operand_loop_dim.items() if relevance['pr'] != {}}
    pr_operand_list = list(pr_operand_loop_LUT.keys())
    mapping_dict_reform = pickle_deepcopy(mapping_dict)
    ''' current and below level pr data size '''
    cabl_pr_data_size = {} #对应分裂出来的r的部分
    ''' current and below level pr data reuse '''
    cabl_pr_data_reuse = {} #对应分裂出来的ir的部分
    ''' each single pr loop data size '''
    per_pr_data_size = {}
    ''' each single pr loop data reuse '''
    per_pr_data_reuse = {}

    for operand in pr_operand_list:
        ''' initialize current and below level pr loop size '''
        cabl_pr_lp_size = {
            pr_data_dim: {
                pr_loop_dim: 1
                for pr_loop_dim in pr_operand_loop_LUT[operand][pr_data_dim]
            }
            for pr_data_dim in pr_operand_loop_LUT[operand].keys()
        } # cabl_pr_lp_size = {'IX': {'OX': 1, 'FX': 1}, 'IY': {'OY': 1, 'FY': 1}}
        ''' initialize current and below level pr data size '''
        cabl_pr_data_size[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))] # mapping_dict[operand]是每个操作数在上一步依次填充进内存层次结构的信息，这里其实就是操作数I的内存层级
            for pr_data_dim in pr_operand_loop_LUT[operand].keys()
        } # cabl_pr_data_size = {'I' : {'IX': [[], [], []], 'IY': [[], [], []]}}
        ''' initialize current and below level pr data reuse '''
        cabl_pr_data_reuse[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))]
            for pr_data_dim in pr_operand_loop_LUT[operand].keys()
        } # cabl_pr_data_reuse = {'I' : {'IX': [[], [], []], 'IY': [[], [], []]}}
        ''' initialize per pr loop data size '''
        per_pr_data_size[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))]
            for pr_data_dim in pr_operand_loop_LUT[operand].keys()
        } # per_pr_data_size = {'I' : {'IX': [[], [], []], 'IY': [[], [], []]}}
        ''' initialize per pr loop data reuse '''
        per_pr_data_reuse[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))]
            for pr_data_dim in pr_operand_loop_LUT[operand].keys()
        } # per_pr_data_reuse = {'I' : {'IX': [[], [], []], 'IY': [[], [], []]}}
        ''' update the cabl_pr_lp_size by multiply pr loop size across architectural level '''
        # 找I的每个内存层级
        for level, loop_list in enumerate(mapping_dict[operand]):
            # 遍历每个内存层级的loop    
            for loop_type, loop_size in loop_list:
                # 如果是r或ir loop，则跳过
                if loop_type in r_ir_operand_loop_LUT[operand]:
                    continue
                # 七个维度里没有IX IY只有OX OY，loop_type只可能是OX或OY，直接遍历pr_operand_loop_LUT里的IX IY,pr_data_dim只能是IX IY
                for pr_data_dim in pr_operand_loop_LUT[operand].keys():
                    # 去找一找当前loop_type是否在pr_operand_loop_LUT[operand][pr_data_dim]中（OY FY）还是（OX FX）反正就一个
                    if any(lp_type == loop_type for lp_type in pr_operand_loop_LUT[operand][pr_data_dim]):
                        cabl_pr_lp_size[pr_data_dim][loop_type] *= loop_size
                        ''' compute pr related data dimension size and data dimension reuse at current and below joint levels
                        based on pr_funcs (dynamic functions extracted in LayerNode). Each pr loop is decoupled into r and ir loops. '''
                        # 找到最小的IX或IY的大小，因为有pr复用，r相关的维度变小了，出来了一些可以复用的ir部分
                        pr_loop_combined_to_r = layer_node.calc_tensor_dim_fraction(operand, cabl_pr_lp_size[pr_data_dim], pr_data_dim)
                        pr_loop_combined_to_ir = prod(cabl_pr_lp_size[pr_data_dim].values()) / pr_loop_combined_to_r
                        cabl_pr_data_size[operand][pr_data_dim][level].append(pr_loop_combined_to_r)
                        cabl_pr_data_reuse[operand][pr_data_dim][level].append(pr_loop_combined_to_ir)
        ''' compute pr related data dimension size and data dimension reuse at each level for each pr loop
         based on cabl_pr_data_size/cabl_pr_data_reuse '''
        for pr_data_dim in cabl_pr_data_size[operand].keys():
            data_size_list = cabl_pr_data_size[operand][pr_data_dim]
            data_reuse_list = cabl_pr_data_reuse[operand][pr_data_dim]
            previous_data_size = 1
            previous_data_data_reuse = 1
            # 算出来r 和 ir的大小是不准确的，上一级的reuse大小要除下一级的reuse大小，才是真正的reuse大小
            for level, va_list in enumerate(data_size_list):
                for idx in range(len(va_list)):
                    per_pr_data_size[operand][pr_data_dim][level].append(data_size_list[level][idx] / previous_data_size)
                    per_pr_data_reuse[operand][pr_data_dim][level].append(data_reuse_list[level][idx] / previous_data_data_reuse)
                    previous_data_size = data_size_list[level][idx]
                    previous_data_data_reuse = data_reuse_list[level][idx]
        # 最后真正用到的就是 per_pr_data_size 和 per_pr_data_reuse，在replace_pr_loop_in_mapping函数中将现有的mapping_dict[operand]中的pr loop根据r ir分裂结果替换为r和ir loop
        mapping_dict_reform[operand] = replace_pr_loop_in_mapping(mapping_dict[operand], per_pr_data_size[operand], per_pr_data_reuse[operand],
                                                                  pr_operand_loop_LUT[operand], r_ir_operand_loop_LUT[operand])

    # return mapping_dict_reform, cabl_pr_data_size, cabl_pr_data_reuse, per_pr_data_size, per_pr_data_reuse
    mapping_dict_reform = {  # 在 decouple pr loop 时，可能会出现大小为 1 的loop，为了不影响后续计算，应排除掉那些大小为 1 的loop
        op: [[item for item in subli if item[1] != 1] for subli in li]
        for op, li in mapping_dict_reform.items()
    }
    return mapping_dict_reform


def replace_pr_loop_in_mapping(single_operand_mapping: Dict, per_pr_data_size: Dict, per_pr_data_reuse: Dict, pr_operand_loop_LUT: Dict,
                               r_ir_operand_loop_LUT: List):
    """
    This function replaces all pr loops in a mapping of a single operand with r and ir loops.
    """
    mapping_new = pickle_deepcopy(single_operand_mapping)

    for level, loop_list in enumerate(single_operand_mapping):
        ''' Introduce the current level pr loop index to distinguish different pr loops at the same architectural level '''
        cl_pr_lp_idx_local = {pr_data_dim: 0 for pr_data_dim in pr_operand_loop_LUT.keys()}
        cl_pr_lp_idx_global = 0
        for idx, (loop_type, loop_size) in enumerate(loop_list):
            if loop_type in r_ir_operand_loop_LUT:
                continue
            for pr_data_dim in pr_operand_loop_LUT.keys():
                if any(lp_type == loop_type for lp_type in pr_operand_loop_LUT[pr_data_dim]):
                    ''' replace the pr loop in the mapping by r loop '''
                    pr_idx_local = cl_pr_lp_idx_local[pr_data_dim]
                    pr_idx_global = cl_pr_lp_idx_global
                    mapping_new[level][idx + pr_idx_global] = \
                        (pr_data_dim + '_r', per_pr_data_size[pr_data_dim][level][pr_idx_local])
                    ''' insert ir loop after the r loop '''
                    # NOTE: Here we insert the ir loop after/above the r loop, which indicates that we ignore the input FIFO effect
                    # during current level feeds data to below level. We could also insert the ir loop before/below the r loop,
                    # which leads to more energy-efficient mapping if the innermost ir loop merging down is enabled.
                    mapping_new[level].insert(idx + pr_idx_global + 1, (pr_data_dim + '_ir', per_pr_data_reuse[pr_data_dim][level][pr_idx_local]))
                    ''' update the pr loop index '''
                    cl_pr_lp_idx_local[pr_data_dim] += 1
                    cl_pr_lp_idx_global += 1

    return mapping_new


def calc_data_size_MAC_count_per_loop(mapping_dict_reform: Dict, operand_loop_dim_reform: Dict):
    """
    This function generates detailed information for each single loop item for each operand.
    """
    detailed_mapping_dict = deepcopy(mapping_dict_reform)
    for operand, mapping_list in mapping_dict_reform.items():
        MAC_count = 1
        data_elem = 1
        for level, loop_list in enumerate(mapping_dict_reform[operand]):
            for idx, (loop_type, loop_size) in enumerate(loop_list):
                MAC_count *= loop_size
                if loop_type in operand_loop_dim_reform[operand]['r']:
                    data_elem *= loop_size
                detailed_mapping_dict[operand][level][idx] = \
                    Loop((loop_type, loop_size), round(MAC_count), round(data_elem))
    return detailed_mapping_dict


if __name__ == "__main__":
    from npuperf.classes.workload.layer_node import LayerNode
    from npuperf.inputs.WL_fromjson.Meta_prototype.workload_inceptionv1 import workload
    test_id = 8
    layer_node = LayerNode(test_id, workload[test_id])

    # # mapping case 1 for debug
    # mapping_dic = {'W': [[], [('FX', 3), ('OX', 3)], [('OX', 7), ('FX', 3)], [('OX', 2)]],
    #                'I': [[('FX', 3)], [('OX', 3)], [('OX', 7), ('FX', 3)], [('OX', 2)]],
    #                'O': [[('FX', 3)], [('OX', 3)], [('OX', 7), ('FX', 3), ('OX', 2)], []]}

    # mapping case 2 for debug
    # mapping_dic = {'W': [[('K', 4)], [('FX', 3), ('OX', 3), ('C', 2)], [('OX', 7), ('FX', 3), ('OY', 3)], [('OX', 2), ('B', 2)]],
    #                'I': [[('K', 4), ('FX', 3)], [('OX', 3), ('C', 2)], [('OX', 7), ('FX', 3), ('OY', 3)], [('OX', 2), ('B', 2)]],
    #                'O': [[('K', 4), ('FX', 3)], [('OX', 3), ('C', 2)], [('OX', 7), ('FX', 3), ('OY', 3), ('OX', 2)], [('B', 2)]]}
    mapping_dic = {
        "O": [[('C', 32.0)], [('C', 6), ('K', 8.0), ('OX', 1.9285714285714286), ('OY', 1.9285714285714286)],
              [('OY', 7), ('OY', 2), ('K', 2), ('OX', 7), ('OX', 2)], []],
        "W": [[('OX', 1.9285714285714286), ('OY', 1.9285714285714286)], [('K', 8.0), ('C', 32.0)],
              [('C', 6), ('OY', 7), ('OY', 2), ('K', 2), ('OX', 7), ('OX', 2)], []],
        "I": [[('K', 8.0), ('OX', 1.9285714285714286), ('C', 32.0), ('OY', 1.9285714285714286)], [('C', 6), ('OY', 7), ('OY', 2), ('K', 2)],
              [('OX', 7), ('OX', 2)]]
    }

    mapping_dict_reform = decouple_pr_loop(mapping_dic, layer_node)
    print('mapping_dict_reform [I]=', mapping_dict_reform['I'])

    # mapping_dict_reform, cabl_pr_data_size, cabl_pr_data_reuse, per_pr_data_size, per_pr_data_reuse = \
    #     decouple_pr_loop(mapping_dic, operand_loop_dim, pr_funcs)
    # print('mapping_dict_reform [I]', mapping_dict_reform['I'])
    # print('cabl_pr_data_size', cabl_pr_data_size)
    # print('cabl_pr_data_reuse', cabl_pr_data_reuse)
    # print('per_pr_data_size', per_pr_data_size)
    # print('per_pr_data_reuse', per_pr_data_reuse)

    # operand_loop_dim_reform = layer_node.operand_loop_dim_reform
    # detailed_mapping_dict = calc_data_size_MAC_count_per_loop(mapping_dict_reform, operand_loop_dim_reform)
    # print('detailed_mapping_dict', detailed_mapping_dict)

# OX = 42
# FX = 9
# IX = 50
# MAC = 42*9 = 378
# reuse = 378/50 = 7.56
#                                                 TOTAL                                      PER LOOP
# a1 = {'I': [[('FX', 3)],            ----------> Data: 3; MAC: 3; Data_Reuse: 1;          | Data: 3; MAC: 3; Data_Reuse: 1;
#             [('OX', 3)],            ----------> Data: 5; MAC: 9; Data_Reuse: 1.8;        | Data: 5/3; MAC: 3; Data_Reuse: 1.8;
#             [('OX', 7), ('FX', 3)], ----------> Data: 23; MAC: 63; Data_Reuse: 2.739;    | Data: 23/5; MAC: 7; Data_Reuse: 1.52;
#                                     ----------> Data: 29; MAC: 189; Data_Reuse: 6.51724; | Data: 29/23; MAC: 3; Data_Reuse: 2.379;
#             [('OX', 2)]]}           ----------> Data: 50; MAC: 378; Data_Reuse: 7.56;    | Data: 50/29; MAC: 2; Data_Reuse: 1.16;
#
# a2 = {'I': [[('IX_r', 3), ('IX_ir', 1)],
#             [('IX_r', 5/3), ('IX_ir', 9/5)],
#             [('IX_r', 23/5), ('IX_ir', (63/23)/(9/5)),
#              ('IX_r', 29/23), ('IX_ir', (189/29)/(63/23))],
#             [('IX_r', 50/29), ('IX_ir', (378/50)/(189/29))]]}
#
# a3 = {'I': [[('IX_r', 3), ('IX_ir', 1)],
#             [('IX_r', 1.67), ('IX_ir', 1.8)],
#             [('IX_r', 4.6), ('IX_ir', 1.52),
#              ('IX_r', 1.26), ('IX_ir', 2.379)],
#             [('IX_r', 1.724), ('IX_ir', 1.16)]]}
