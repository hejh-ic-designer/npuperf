"""
This file contains the core code of the temporal mapping optimization method
called loma: loop order based memory allocation.

TODO: Get a layers' dimensions to generate the multiset permutations for all loop types
TODO: Write generator that takes loop-type-specific multiset permutations and generates loop order permutation
TODO: Write uneven memory allocator, that allocates the loops of the loop order bottom-up to the memories in the hierarchy
TODO: (optional) Write even memory allocator
TODO: Once we have allocated the loops to the different hierarchy levels, call the cost model to get energy, latency
TODO: Save the best found loop order (and its associated allocated mapping)
"""
import operator
import numpy as np
from sympy.ntheory import factorint
import logging
from npuperf.classes.workload.layer_node import LayerNode
from npuperf.classes.mapping.spatial.spatial_mapping import SpatialMapping
from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from npuperf.classes.opt.temporal.loma.multipermute import permutations
from npuperf.classes.opt.temporal.loma.memory_allocator import MemHierarchyTooSmallException, MemoryAllocator
from math import factorial, prod

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class LomaEngine_Stationary:
    """
    Class that handles optimization of temporal mapping given a:
    - layer
    - spatial mapping
    - a memory hierarchy
    This optimization is carried out through loop order based memory allocation.
    For each ordering of the temporal loops, they are allocated bottom-up to the
    levels in the memory hierarchy.
    See https://ieeexplore.ieee.org/document/9458493 for more details.
    """

    def __init__(self, *, accelerator, layer, spatial_mapping, stationary, loma_lpf_limit=np.inf,  **kwargs):
        """
        Initialize the engine with the given:
        - Accelerator
        - LayerNode
        - SpatialMapping

        The memory hierarchy from the correct core is extracted from the accelerator.
        :param accelerator: accelerator to use the memory hierarchy of
        :param layer: layer to generate temporal mappings for
        :param spatial_mapping: SpatialMapping to use
        :param loma_lpf_limit:
        :param kwargs: further unused, for ease of calling only
        """
        self.lpf_limit = loma_lpf_limit

        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.stationary = stationary      # 'O', 'W', 'I'
        # 循环相关性原理，见zigzag论文Fig.4 & Fig.5: https://ieeexplore.ieee.org/document/9360462
        self.stationary_relevant = {'O': ['B', 'K', 'OY', 'OX'], 'W': ['K', 'C', 'FY', 'FX']}

        if stationary == 'I':
            # 如果是 input stationary, 则更新关于 I 的相关性， 其原理在zigzag 论文Fig.5 处
            # input stationary分为三种情况，但本质上都是需要在4个间接相关的维度中取出两个相关的维度
            # 先检测spatial mapping，如果有kernel 宽高维度的展开或OFM 宽高维度的展开，则取出这一个或两个维度作为相关的
            # 然后，若spatial mapping 中取出的维度不足两个（即1个或0个），那么从考虑时间上的维度，从'FX', 'FY', 'OX', 'OY'中取出合理的维度构成时间上相关的维度
            # 这里考虑时间上相关的维度，需要考虑具体的硬件卷积映射的方式，例如Envision 是'FX' 时间维度和'OX' 空间维度构成了 input stationary
            temporal_dim_for_I_relevant = ['FX', 'OX']      #! 如果是input stationary，这个值可能需要从外部更改. 如果是两个值，应该均为Y 或均为X，而且一个是F，一个是O
            I_r1, I_r2 = self.set_input_stationary(self.layer.user_spatial_mapping, temporal_dim_for_I_relevant)
            logger.info(f'For input stationary, the relevant dim of input is {I_r1} and {I_r2}')
            self.stationary_relevant = {'O': ['B', 'K', 'OY', 'OX'], 'W': ['K', 'C', 'FY', 'FX'], 'I': ['B', 'C', I_r1, I_r2]}

        # Extract the memory hierarchy from the accelerator
        # TODO: Take into account that data might be stored in lower level,
        # TODO: thus adapt the memory hierarchy.
        # TODO: The fact that there is a global buffer above the cores requires attention.
        core_id = layer.core_allocation
        self.memory_hierarchy: MemoryHierarchy = accelerator.get_core(core_id).memory_hierarchy

    def set_input_stationary(self, usr_spm: dict, temporal_dim_for_I_relevant: list):
        # 返回的两个值，应该是一个F，一个O，而且均为X 或均为Y
        # usr_spm: user spatial mapping, e.g., {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)}
        # temporal_dim_for_I_relevant: 'FX', 'FY', 'OX', 'OY'中的一个或两个
        spatial_unroll_dim = [unroll[0] for unroll in usr_spm.values()]     # 取出spatial mapping 中的所有维度，e.g., ['K', 'C', 'FY', 'OY']
        target_list = ['OY', 'OX', 'FY', 'FX']
        spatial_relevant = list(set(spatial_unroll_dim).intersection(set(target_list)))   # 两者的交集, 可能有 0，1，2 个元素

        if len(spatial_relevant) == 2:  # 空间上的对IFM 的reuse
            # 这里应该先对spatial_relevant 进行检查，符合规则：两个值应该是一个F，一个O，而且均为X 或均为Y
            assert spatial_relevant in [['FY', 'OY'], ['OY', 'FY'], ['FX', 'OX'], ['OX', 'FX']], f"Input stationary error! spatial relevant error, {spatial_relevant}"
            return spatial_relevant

        assert (len(spatial_relevant) + len(temporal_dim_for_I_relevant) == 2), f"Input stationary set error!, spatial relevant is {spatial_relevant}, temporal relevant is {temporal_dim_for_I_relevant}. \
            The sum of the number of both elements should be equal to 2, please check parameter 'temporal_dim_for_I_relevant' "

        if len(spatial_relevant) == 0:  # 时间上的对IFM 的reuse
            return temporal_dim_for_I_relevant

        spatio_temporal_relevant = spatial_relevant + temporal_dim_for_I_relevant
        assert spatio_temporal_relevant in [['FY', 'OY'], ['OY', 'FY'], ['FX', 'OX'], ['OX', 'FX']], f"Input stationary error! spatio temporal relevant error, {spatio_temporal_relevant}"
        return spatio_temporal_relevant       # 时空上的对IFM 的reuse

    def run(self):
        """
        :returns : Generator that yields all temporal mappings
        TODO: add the criterion(s) as inputs to this function.
        """
        self.get_temporal_loops()  # get all the temporal loops
        self.get_prime_factors()  # convert these to LPFs (loop prime factors)
        self.set_stationary()
        yielded = False

        counts = self.temporal_loop_pf_counts
        # total = factorial(sum(sum(v) for v in counts.values())) / prod(factorial(s) for s in sum(counts.values(), tuple()))     # 总的temoporal mapping 数量
        total = factorial(len(self.lpf_ir)) * factorial(len(self.lpf_r)) / prod(factorial(s) for s in sum(counts.values(), tuple()))     # 总的temoporal mapping 数量
        logger.debug(f"Loma engine will run {total}")
        printsize = total // 10
        self.count = 0

        logger.debug(f'lpfs: {self.lpfs}')
        logger.debug(f'lpf_ir: {self.lpf_ir}')
        logger.debug(f'lpf_r: {self.lpf_r}')

        if self.lpf_ir == [] or self.lpf_r == []:   # 有一个是空列表的话，就对整个的lpfs 排序
            for ordering in self.og(self.lpfs):
                allocator = MemoryAllocator(self.accelerator, self.layer, self.spatial_mapping, ordering)
                try:
                    temporal_mapping = allocator.run()  # allocate this ordering to the memories
                    yielded = True
                    yield temporal_mapping
                    if total > 10000 and self.count % printsize == 0:
                        logger.debug(f"Loma engine ran {self.count} of {total}")
                except MemHierarchyTooSmallException:
                    pass
                self.count += 1

            if not yielded:
                raise MemHierarchyTooSmallException("No loop ordering was found that did not exceed memory capacity")
        elif self.lpf_ir == [] and self.lpf_r == []:
            raise ValueError(f'lpf_ir and lpf_r are both empty list, self.lpfs is {self.lpfs}')
        else:   #* 一般的情况
            for ir_ordering in self.og(self.lpf_ir):
                for r_ordering in self.og(self.lpf_r):
                    #* ir 的在内侧，r 的在外侧
                    ordering:list = ir_ordering + r_ordering
                    allocator = MemoryAllocator(self.accelerator, self.layer, self.spatial_mapping, ordering)
                    try:
                        temporal_mapping = allocator.run()  # allocate this ordering to the memories
                        yielded = True
                        yield temporal_mapping
                        if total > 10000 and self.count % printsize == 0:
                            logger.debug(f"Loma engine ran {self.count} of {total}")
                    except MemHierarchyTooSmallException:
                        pass
                    self.count += 1

            if not yielded:
                raise MemHierarchyTooSmallException("No loop ordering was found that did not exceed memory capacity")


    def get_temporal_loops(self):
        """
        Get all loops that have to be temporally scheduled given layer and spatial mapping.
        先取得空间展开的值, 一个例子是 [('OY', 4.0), ('OX', 4.0), ('K', 32.0), ('C', 2.0)],  然后用layer 总的loop size除以空间上的展开, 得到时间上的loop size
        """
        temporal_loop_dim_size = self.layer.loop_dim_size.copy()  # init with all loop sizes
        for spatial_loop in self.spatial_mapping.spatial_loop_dim_size:
            (spatial_loop_dim, spatial_loop_size) = spatial_loop
            # Allow greedy mapping. If the spatial unrolling is not a multiple of the layer dimension size,
            # we take the ceil of the division, so there can be one extra temporal iteration.
            q = int(np.ceil(temporal_loop_dim_size[spatial_loop_dim] / spatial_loop_size))
            # q, rem = divmod(temporal_loop_dim_size[spatial_loop_dim], spatial_loop_size)
            # assert rem == 0, "Division of dimension size by spatial unrolling size is not an integer"
            if q == 1:
                del temporal_loop_dim_size[spatial_loop_dim]        # 对于存在空间展开的维度，如果相除后等于1，那么应该删掉这个维度
            else:
                temporal_loop_dim_size[spatial_loop_dim] = q
        self.temporal_loop_dim_size = temporal_loop_dim_size

    def get_prime_factors(self):
        """
        Get the prime factors for all temporal loops.

        This is saved in three separate class attributes:
        temporal_loop_pfs: a dict that for each temporal loop dimension contains the prime factors
        temporal_loop_pf_counts: a dict that for each temporal loop dimension contains the prime factor multiplicities
        temporal_loop_pf_count_sums: a dict that for each temporal loop dimension contains the total amount of prime factors
        """
        temporal_loop_pfs = {}
        temporal_loop_pf_counts = {}
        temporal_loop_pf_count_sums = {}
        lpfs = []
        for (tl_dim, tl_size) in self.temporal_loop_dim_size.items():  # tl = temporal loop
            factors = factorint(tl_size)    # 质因数分解
            pfs = []
            counts = []
            for pf, multiplicity in factors.items():
                pfs.append(pf)
                counts.append(multiplicity)
                for i in range(multiplicity):
                    lpfs.append((tl_dim, pf))
            temporal_loop_pfs[tl_dim] = tuple(pfs)
            temporal_loop_pf_counts[tl_dim] = tuple(counts)
            temporal_loop_pf_count_sums[tl_dim] = sum(counts)

        logger.info(f"Generated {len(lpfs)} LPFs for layer {self.layer}.")

        self.temporal_loop_pfs = temporal_loop_pfs
        self.temporal_loop_pf_counts = temporal_loop_pf_counts
        self.temporal_loop_pf_count_sums = temporal_loop_pf_count_sums
        self.lpfs = lpfs
        # logger.info(f'before limit, the lpfs is: {self.lpfs}')
        # Limit the number of lpfs (if this is set in the settings)
        self.limit_lpfs()

    def limit_lpfs(self):
        """
        Function to limit the total number of loop prime factors present in this instance.
        This function scans the lpfs and while the number of lpfs is greater than self.lpf_limit it:
        - picks the loop dimension that has the most lpfs
        - merges the smallest two lpfs of that loop dimension (multiplying their values)
        """
        n_pf = sum(self.temporal_loop_pf_count_sums.values())
        if n_pf <= self.lpf_limit:
            logger.info(f"No lpf limiting performed for layer {self.layer}")
            return
        while n_pf > self.lpf_limit:
            # Find the loop dimension with the most lpfs
            max_ld = max(self.temporal_loop_pf_count_sums.items(), key=operator.itemgetter(1))[0]
            # Get the prime factors of this loop dimension
            max_pfs = list(self.temporal_loop_pfs[max_ld])
            # Get the multiplicity of these prime factors
            max_counts = list(self.temporal_loop_pf_counts[max_ld])

            if max_counts[0] == 1:  # multiplicity of smallest pf is 1
                new_factor = max_pfs[0] * max_pfs[1]
                max_counts[0] -= 1
                max_counts[1] -= 1
            else:  # multiplicity of smalles pf is > 1
                new_factor = max_pfs[0] * max_pfs[0]
                max_counts[0] -= 2

            if new_factor in max_pfs:  # possible if not first iteration of while loop
                new_factor_idx = max_pfs.index(new_factor)
                max_counts[new_factor_idx] += 1
            else:  # the new factor is not yet present in the factors, insert so list remains sorted
                new_factor_idx = len([pf for pf in max_pfs if pf < new_factor])
                max_pfs.insert(new_factor_idx, new_factor)
                max_counts.insert(new_factor_idx, 1)  # first time this factor occured, count = 1

            # Sanitize max_pfs and max_counts to remove all elements with multiplicity 0
            non_zero_idxs = [idx for idx, count in enumerate(max_counts) if count != 0]
            max_pfs = [max_pfs[non_zero_idx] for non_zero_idx in non_zero_idxs]
            max_counts = [max_counts[non_zero_idx] for non_zero_idx in non_zero_idxs]

            # Update the appropriate variables with these new factors and multiplicities
            self.temporal_loop_pfs[max_ld] = tuple(max_pfs)
            self.temporal_loop_pf_counts[max_ld] = tuple(max_counts)
            self.temporal_loop_pf_count_sums[max_ld] -= 1

            # Decrease the total number of factors by 1
            n_pf -= 1

        # Update self.lpfs for these new factors
        lpfs = []
        for dim in self.temporal_loop_pfs.keys():
            for (pf, count) in zip(self.temporal_loop_pfs[dim], self.temporal_loop_pf_counts[dim]):
                lpfs += list(((dim, pf),) * count)
        self.lpfs = lpfs

        # logger.info(f"Limited layer {self.layer} to {len(self.lpfs)} lpfs. And then lpfs is {self.lpfs}")  # 显示所有的 lpfs
        return

    def set_stationary(self):
        # 把相关的和不相关的维度提取出来，分别进行排序，然后再合并在一起
        self.lpf_r = [r_tuple for r_tuple in self.lpfs if r_tuple[0] in self.stationary_relevant[self.stationary]]
        self.lpf_ir = [ir_tuple for ir_tuple in self.lpfs if ir_tuple[0] not in self.stationary_relevant[self.stationary]]
        logger.info(f'set layer {self.layer} to {self.stationary} stationary.')
        if self.lpf_ir == [] or self.lpf_r == []:
            logger.warning(f'Sorting an empty list may cause errors. better to check')

    def og(self, some_lpf):
        """
        Generator that yields all orderings of the temporal loops.
        """
        # The lpfs are stored in self.lpfs
        return permutations(some_lpf)


if __name__ == "__main__":
    from npuperf.inputs.WL_fromjson.Meta_prototype.workload_mv1 import workload
    from classes.hardware.architecture.memory_hierarchy import memory_hierarchy_example1
    from classes.hardware.architecture.operational_array import multiplier_array_example1


    layer = workload[1]
    layer_node = LayerNode(layer)

    '''This dictionary should be generated automatically from the user provided spatial mapping if there is one'''
    spatial_mapping_dic = {'W': [[('OX', 14), ('OY', 7)], [('FX', 3)], [], []],
                           'I': [[], [('OX', 14), ('FX', 3)], [('OY', 7)], []],
                           'O': [[('FX', 3)], [('OX', 14), ('OY', 7)], []]}

    spatial_mapping = SpatialMapping(spatial_mapping_dict=spatial_mapping_dic, layer_node=layer_node)

    multiplier_array = multiplier_array_example1()
    memory_hierarchy = memory_hierarchy_example1(multiplier_array)

    loma_engine = LomaEngine_Stationary(layer=layer_node,
                                    spatial_mapping=spatial_mapping,
                                    memory_hierarchy=memory_hierarchy)
    loma_engine.run()
    a = 1


    # 对于这样的 temporal loop dim size: (即 原本的 layer dim size 除去了 spatial unroll)
    {'B': 1, 'C': 32, 'OY': 20, 'OX': 20, 'FY': 3, 'FX': 3}

    # temporal_loop_pfs: 找出质因数
    {'B': (), 'C': (2,), 'OY': (2, 5), 'OX': (2, 5), 'FY': (3,), 'FX': (3,)}
    # temporal_loop_pf_counts: 找出对应质因数的个数
    {'B': (), 'C': (5,), 'OY': (2, 1), 'OX': (2, 1), 'FY': (1,), 'FX': (1,)}
    # temporal_loop_pf_count_sums: 所有质因数的个数之和
    {'B': 0, 'C': 5, 'OY': 3, 'OX': 3, 'FY': 1, 'FX': 1}
    # lpfs: 分解为最细粒度的循环嵌套
    [('C', 2), ('C', 2), ('C', 2), ('C', 2), ('C', 2), ('OY', 2), ('OY', 2), ('OY', 5), ('OX', 2), ('OX', 2), ('OX', 5), ('FY', 3), ('FX', 3)]
    # After lpf limit: (set lpf_limit = 6) 根据limit 进行合并
    [('C', 32), ('OY', 20), ('OX', 4), ('OX', 5), ('FY', 3), ('FX', 3)]