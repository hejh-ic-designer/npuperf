from collections import defaultdict
from typing import Set, Tuple, List, Dict
import networkx as nx
from networkx import DiGraph

from npuperf.classes.hardware.architecture.memory_instance import MemoryInstance
from npuperf.classes.hardware.architecture.memory_level import MemoryLevel
from npuperf.classes.hardware.architecture.operational_array import OperationalArray


class MemoryHierarchy(DiGraph):
    """
    用有向图表示的内存层次结构, 从最低级别的mem(reg) 到 最高级别的mem(DRAM) 指向    \\
    Class that represents a memory hierarchy as a directed networkx graph.  \\
    The memory hierarchy graph is directed, with the root nodes representing the lowest level in the memory hierarchy.
    """
    def __init__(self, operational_array: OperationalArray, name: str = "Memory Hierarchy", **attr):
        """
        Initialize the memory hierarchy graph.
        The initialization sets the operational array this memory hierarchy will connect to.
        The graph nodes are the given nodes. The edges are extracted from the operands the memory levels store.
        :param nodes: a list of MemoryLevels. Entries need to be provided from lowest to highest memory level.
        """
        super().__init__(**attr)
        self.name = name
        self.operational_array = operational_array

        # Initialize the set that will store all memory operands
        # 很简单，{'I1', 'O', 'I2'}
        self.operands: Set[str] = set()

        # Initialize the dict that will store how many memory levels an operand has
        # 就是一个操作数拥有多少个mem lv, 例如metaprototype的nb_levels：{'I2': 4, 'O': 4, 'I1': 3}
        self.nb_levels: Dict[str, int] = {}

        self.mem_instance_list: List[MemoryLevel] = []

    def __jsonrepr__(self):
        """
        JSON Representation of this object to save it to a json file.
        """
        return {"memory_levels": [node for node in nx.topological_sort(self)]}

    def add_memory(self, memory_instance: MemoryInstance, operands: Tuple[str, ...], port_alloc: Tuple[dict, ...],
                   served_dimensions: Set or str):
        """
        Adds a memory to the memory hierarchy graph.

        !!! NOTE: memory level need to be added from bottom level (e.g., Reg) to top level (e.g., DRAM) for each operand !!!

        Internally a MemoryLevel object is built, which represents the memory node.
        Edges are added from all sink nodes in the graph to this node if the memory operands match
        :param memory_instance: The MemoryInstance containing the different memory characteristics.
        :param operands: The memory operands the memory level stores.
        :param served_dimensions: The operational array dimensions this memory level serves.
        served_dimensions: 此内存级别所服务的PE array维度   \\
        集合中的每个向量都是所服务的方向。  \\
        使用“all”表示所有维度, 即内存级别未展开 \\
        Each vector in the set is a direction that is served.
        Use 'all' to represent all dimensions (i.e. the memory level is not unrolled).
        """
        # Assert that if served_dimensions is a string, it is "all"
        if type(served_dimensions) == str:
            assert served_dimensions == "all", "Served dimensions is a string, but is not all."

        # Add the memory operands to the self.operands set attribute that stores all memory operands.
        for mem_op in operands:
            if mem_op not in self.operands:
                self.nb_levels[mem_op] = 1
                self.operands.add(mem_op)
            else:
                self.nb_levels[mem_op] += 1
            self.operands.add(mem_op)

        # Parse the served_dimensions by replicating it into a tuple for each memory operand
        # as the MemoryLevel constructor expects this.
        served_dimensions_repl = tuple([served_dimensions for _ in range(len(operands))])
        # 这里是把served dimensions重复operands的个数遍，并且使其成为一个tuple
        # e.g.: operands=('I2','I1)     served_dimensions='all'     那么就是（'all', 'all')
        # e.g.: operands=('I2','I1)     served_dimensions={(0, 0, 1, 0),(0, 1, 0, 0)}     那么就是 ({(0, 1, 0, 0), (0, 0, 1, 0)}, {(0, 1, 0, 0), (0, 0, 1, 0)})

        # Compute which memory level this is for all the operands
        mem_level_of_operands = {}
        for operand in operands:
            nb_levels_so_far = len([node for node in self.nodes() if operand in node.operands])
            mem_level_of_operands[operand] = nb_levels_so_far

        #: mem_level_of_operands 变量说明：这是MemoryLevel的入参
        # 以Meta_prototype_DF为例，每一次add memory，会把一个mem instance加进来到mem hier，加了7片，就有7个mem_level_of_operands，分配7次MemoryLevel的时候，就会传入对应的这个参数
        # 他的意思是，当前加入的mem level所分配的操作数是哪些，后面的数字代表对应操作数的级别
        # 因为添加的时候，必须从最低级到最高级添加，那么比如到最后一个DRAM的时候
        # {'I1': 2, 'I2': 3, 'O': 3}代表了，DRAM可以存放I1,I2和O，同时它的I1是第2级，I2和O是第3级（注意这里是从第0级开始算的）
        # 需要区分的是，MemoryHierarchy自带了一个 nb_levels 变量，这是整个hier的所有操作数的级别个数，例如：{'I2': 4, 'O': 4, 'I1': 3}
        # mem_level_of_operands 是MemoryLevel的入参，而nb_levels是mem hier图的self变量

        # mem_level_of_operands:  {'I2': 0}
        # mem_level_of_operands:  {'O': 0}
        # mem_level_of_operands:  {'I2': 1}
        # mem_level_of_operands:  {'I2': 2}
        # mem_level_of_operands:  {'I1': 0, 'O': 1}
        # mem_level_of_operands:  {'I1': 1, 'O': 2}
        # mem_level_of_operands:  {'I1': 2, 'I2': 3, 'O': 3}

        memory_level = MemoryLevel(memory_instance=memory_instance,
                                   operands=operands,
                                   mem_level_of_operands=mem_level_of_operands,
                                   port_alloc=port_alloc,
                                   served_dimensions=served_dimensions_repl,
                                   operational_array=self.operational_array)

        self.mem_instance_list.append(memory_level)

        # Precompute appropriate edges
        to_edge_from = set()
        for mem_op in operands:
            # Find top level memories of the operands
            for m in self.get_operator_top_level(mem_op)[0]:
                to_edge_from.add(m)

        # Add the node to the graph
        self.add_node(memory_level)

        for sink_node in to_edge_from:
            # Add an edge from this sink node to the current node
            self.add_edge(sink_node, memory_level)



    #= 一些get 函数，和两个rm 函数
    def get_memory_levels(self, mem_op: str):
        """
        返回的是一个列表, 里面存放了所有拥有mem_op的MemoryLevel 对象    \\
        Returns a list of memories in the memory hierarchy for the memory operand.
        The first entry in the returned list is the innermost memory level.
        """
        # Sort the nodes topologically and filter out all memories that don't store mem_op
        memories = [node for node in nx.topological_sort(self) if mem_op in node.operands]
        return memories

    def get_operands(self):
        """
        Returns all the memory operands this memory hierarchy graph contains as a set.
        """
        return self.operands

    def get_inner_memories(self) -> List[MemoryLevel]:
        """
        返回最低级别的MemoryLevel, 一般是reg级别的, 用入度 == 0 判断  \\
        Returns the inner-most memory levels for all memory operands.
        """
        memories = [node for node, in_degree in self.in_degree() if in_degree == 0]
        return memories

    def get_outer_memories(self) -> List[MemoryLevel]:
        """
        返回最高级别的MemoryLevel, 一般是DRAM级别的, 用出度 == 0 判断   \\
        Returns the outer-most memory levels for all memory operands.
        edge
        :return:
        """
        memories = [node for node, out_degree in self.out_degree() if out_degree == 0]
        return memories

    def get_top_memories(self) -> Tuple[List[MemoryLevel], int]:
        """
        Returns the 'top'-most MemoryLevels, where 'the' level of MemoryLevel is considered to be the largest
        level it has across its assigned operands
        :return: (list_of_memories_on_top_level, top_level)
        """
        level_to_mems = defaultdict(lambda: [] )
        for node in self.nodes():
            level_to_mems[max(node.mem_level_of_operands.values())].append(node)
        top_level = max(level_to_mems.keys())
        return level_to_mems[top_level], top_level

    def remove_top_level(self) -> Tuple[List[MemoryLevel], int]:
        """
        这个remove 是真的把最高级别node 从mem hier中删掉了的    \\
        Removes the top level of this memory hierarchy.
        'The' level of MemoryLevel instance is considered to be the largest level it has across its assigned operands,
        and those with the highest appearing level will be removed from this MemoryHierarchy instance
        :return: (removed_MemoryLevel_instances, new_number_of_levels_in_the_hierarchy)
        """
        to_remove, top_level = self.get_top_memories()
        for tr in to_remove:
            self.mem_instance_list.remove(tr)
            self.remove_node(tr)

        for k in self.nb_levels:
            self.nb_levels[k] = len(set(node.mem_level_of_operands.get(k)
                                        for node in self.nodes() if k in node.mem_level_of_operands))
        return to_remove, max(self.nb_levels.keys())

    def get_operator_top_level(self, operand) -> Tuple[List[MemoryLevel], int]:
        """
        Finds the highest level of memories that have the given operand assigned to it, and returns the MemoryLevel
        instance on this level that have the operand assigned to it.
        'The' level of a MemoryLevel is considered to be the largest
        level it has across its assigned operands.
        :return: (list of MemoryLevel instances, top_level)
        """
        level_to_mems = defaultdict(lambda: [] )
        for node in self.nodes():
            if operand in node.operands[:]:
                level_to_mems[node.mem_level_of_operands[operand]].append(node)
        top_level = max(level_to_mems.keys()) if level_to_mems else -1
        assert len(level_to_mems[top_level]) <= 1, "Don't know what to do with multiple memories at the same level"
        return level_to_mems[top_level], top_level

    def remove_operator_top_level(self, operand):
        """
        Finds the highest level of memories that have the given operand assigned to it, and returns the MemoryLevel
        instance on this level that have the operand assigned to it AFTER removing the operand from its operands.
        'The' level of a MemoryLevel is considered to be the largest
        level it has across its assigned operands.
        If a memory has no operands left, it is removed alltogether.
        :return: (list of MemoryLevel instance that have the operand removed, new top_level of the operand)
        """
        to_remove, top_level = self.get_operator_top_level(operand)

        served_dimensions = []
        for tr in to_remove:
            del tr.mem_level_of_operands[operand]
            tr.operands.remove(operand)
            for p in tr.port_list:
                for so in p.served_op_lv_dir[:]:
                    if so[0] == operand:
                        p.served_op_lv_dir.remove(so)
            if len(tr.mem_level_of_operands) == 0:
                self.mem_instance_list.remove(tr)
                self.remove_node(tr)


        for k in self.nb_levels:
            self.nb_levels[k] = len(set(node.mem_level_of_operands.get(k)
                                        for node in self.nodes() if k in node.mem_level_of_operands))

        return to_remove, self.nb_levels[operand]

    def get_memorylevel_with_id(self, id):
        for n in self.nodes():
            if n.get_id() == id:
                return n
        raise ValueError(f"No memorylevel with id {id} in this memory hierarchy")
