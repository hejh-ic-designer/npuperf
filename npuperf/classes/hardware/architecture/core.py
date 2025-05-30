from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from npuperf.classes.hardware.architecture.operational_array import OperationalArray
    from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
    from npuperf.classes.hardware.architecture.memory_level import MemoryLevel
import networkx as nx

class Core:
    """
    The Core class houses the array of multipliers and the attached memory hierarchy.
    This class supports a singular multiplier array and memory hierarchy, runtime flexibility should be implemented on top.
    """
    def __init__(self, id: int, operational_array: 'OperationalArray', memory_hierarchy: 'MemoryHierarchy', dataflows=None):
        self.id = id
        self.operational_array = operational_array
        self.memory_hierarchy = memory_hierarchy
        self.dataflows = (
            dataflows  # save the possible spatial dataflows inside the Core
        )
        self.check_valid()

        self.recalculate_memory_hierarchy_information()

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return self.__dict__

    def check_valid(self):
        pass

    def recalculate_memory_hierarchy_information(self):
        self.generate_memory_hierarchy_dict()
        self.generate_memory_sharing_list()

    def generate_memory_hierarchy_dict(self):
        mem_operands = self.memory_hierarchy.nb_levels.keys()
        mem_hierarchy_dict = {}
        mem_size_dict = {}
        mem_r_bw_dict = {}
        mem_w_bw_dict = {}
        mem_r_bw_min_dict = {}
        mem_w_bw_min_dict = {}
        mem_r_energy_dict = {}  # only for core_generator test
        mem_w_energy_dict = {}  # only for core_generator test
        for mem_op in mem_operands:
            mem_hierarchy_dict[mem_op] = [node for node in nx.topological_sort(self.memory_hierarchy)
                                          if mem_op in node.operands]
            mem_size_dict[mem_op] = [node.memory_instance.size for node in nx.topological_sort(self.memory_hierarchy)
                                     if mem_op in node.operands]
            mem_r_bw_dict[mem_op] = [node.memory_instance.r_bw for node in nx.topological_sort(self.memory_hierarchy)
                                     if mem_op in node.operands]
            mem_w_bw_dict[mem_op] = [node.memory_instance.w_bw for node in nx.topological_sort(self.memory_hierarchy)
                                     if mem_op in node.operands]
            mem_r_bw_min_dict[mem_op] = [node.memory_instance.r_bw_min for node in nx.topological_sort(self.memory_hierarchy)
                                         if mem_op in node.operands]
            mem_w_bw_min_dict[mem_op] = [node.memory_instance.w_bw_min for node in nx.topological_sort(self.memory_hierarchy)
                                         if mem_op in node.operands]
            mem_r_energy_dict[mem_op] = [node.memory_instance.r_cost for node in nx.topological_sort(self.memory_hierarchy)
                                         if mem_op in node.operands]
            mem_w_energy_dict[mem_op] = [node.memory_instance.w_cost for node in nx.topological_sort(self.memory_hierarchy)
                                         if mem_op in node.operands]
        self.mem_hierarchy_dict = mem_hierarchy_dict
        self.mem_size_dict = mem_size_dict
        self.mem_r_bw_dict = mem_r_bw_dict
        self.mem_w_bw_dict = mem_w_bw_dict
        self.mem_r_bw_min_dict = mem_r_bw_min_dict
        self.mem_w_bw_min_dict = mem_w_bw_min_dict
        self.mem_r_energy_dict = mem_r_energy_dict
        self.mem_w_energy_dict = mem_w_energy_dict

    def generate_memory_sharing_list(self):
        """
        Generates a list of dictionary that indicates which operand's which memory levels are sharing the same physical memory  \\
        以meta_prototype_DF 为例, memory_sharing_list is [{'I1': 2, 'I2': 3, 'O': 3}, {'I1': 0, 'O': 1}, {'I1': 1, 'O': 2}] \\
        即, I1的第2级、I2的第三级、和O的第3级共享了一片物理内存(即DRAM), list中有三项, 即列出了3片物理上有共享关系的内存片
        """
        memory_sharing_list = []
        for mem_lv in self.mem_hierarchy_dict.values():
            for mem in mem_lv:
                operand_mem_share = mem.mem_level_of_operands
                if len(operand_mem_share) > 1 and operand_mem_share not in memory_sharing_list:
                    memory_sharing_list.append(operand_mem_share)

        self.mem_sharing_list = memory_sharing_list

    # 一些get 函数，供外部调用
    def get_memory_hierarchy(self):
        return self.memory_hierarchy

    def get_memory_hierarchy_dict(self):
        return self.mem_hierarchy_dict

    def get_memory_size_dict(self):
        return self.mem_size_dict

    def get_memory_bw_dict(self):
        return self.mem_r_bw_dict, self.mem_w_bw_dict

    def get_memory_bw_min_dict(self):
        return self.mem_r_bw_min_dict, self.mem_w_bw_min_dict

    def get_memory_sharing_list(self):
        return self.mem_sharing_list

    def get_memory_level(self, mem_op: str, mem_lv: int) -> 'MemoryLevel':
        """
        Returns a specific memory level in the memory hierarchy for the memory operand.
        """
        # Sort the nodes topologically and filter out all memories that don't store mem_op
        memory = [node for node in nx.topological_sort(self.memory_hierarchy) if mem_op in node.operands]
        return memory[mem_lv]

    def get_lowest_shared_mem_level_above(self, mem_op1, mem_lv1, mem_op2, mem_lv2):
        """
        Get the lowest shared memory level between mem_op1 (>= mem_lv1) and mem_op2 (>= mem_lv2).
        """
        for lv, mem in enumerate(self.mem_hierarchy_dict[mem_op1][mem_lv1:]):
            if mem_op2 in mem.operands and mem_lv2 <= mem.mem_level_of_operands[mem_op2]:
                return mem

        raise Exception(f"{mem_op1}'s level {mem_lv1} and {mem_op2}'s level {mem_lv2} don't have a shared memory above!")
