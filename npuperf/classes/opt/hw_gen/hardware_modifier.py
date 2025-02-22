import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from npuperf.classes.hardware.architecture.accelerator import Accelerator
from npuperf.classes.hardware.architecture.core import Core
from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from npuperf.classes.hardware.architecture.operational_array import OperationalArray
from npuperf.classes.opt.hw_gen.core_generator import CoreGenerator
from npuperf.utils import pickle_deepcopy

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)


class HardwareModifier:
    """
    use HardwareModifier.from_arg() method to create an object
    
    use get_accelerator() method to return an modified accelerator object
    """

    def __init__(self,
                 base_acc: 'Accelerator',
                 acc_name: str,
                 MAC_counts: str | None,
                 global_buffer_bandwidth: str | None,
                 global_buffer_size: str | None,
                 ddr_bandwidth: str | None,
                 ddr_size=None):
        """ Simply modify a hardware

        Args:
            base_acc (Accelerator): base acc object to modified
            acc_name (str): name
            MAC_counts (int): the number of total MACs or PEs, typically 1024, 2048, etc.
            global_buffer_bandwidth (int): bit/cc, typically 128
            global_buffer_size (int): MB, typically 1 or 2
            ddr_bandwidth (int): bit/cc, typically 96
            ddr_size (int, optional): GB, Defaults to None will means DDR is almost big to Unlimited.
        """
        assert len(base_acc.cores) == 1, f'Only support Single Core Accelerator, current Acc has {len(base_acc.cores)} Cores'
        self.base_acc = base_acc
        self.acc_name = acc_name
        self.MAC_counts_str = MAC_counts
        self.global_buffer_bandwidth_str = global_buffer_bandwidth
        self.global_buffer_size_str = global_buffer_size
        self.ddr_bandwidth = ddr_bandwidth
        self.ddr_size = ddr_size
        self.accelerator = self.modify_hardware()

    def get_accelerator(self):
        return self.accelerator

    @classmethod
    def from_arg(cls, acc: 'Accelerator', config_info: 'Namespace'):
        """接受一个NameSpace 类的模板来修改 acc 实例

        Args:
            acc (Accelerator): 在这个acc 的基础上做修改
            config_info (Namespace): 用户提供的配置参数

        Returns:
            type[Self@HardwareModifier]
        """
        if cls.check_valid(config_info):

            return cls(base_acc=acc,
                       acc_name=config_info.hw + '_user',
                       MAC_counts=config_info.MACs,
                       global_buffer_bandwidth=config_info.gb_bw,
                       global_buffer_size=config_info.gb_size,
                       ddr_bandwidth=config_info.dram_bw,
                       ddr_size=config_info.dram_size if hasattr(config_info, 'dram_size') else None)

    @classmethod
    def check_valid(cls, args: 'Namespace'):
        """对输入的模板进行合法性检查, 如果不合法, 则生成 error_info 抛出
        """
        error_info = {}

        # MACs
        """
        检查MACs的逻辑为它是否是 2 的整数次幂, 因为一般accelerator的MACs都是 2 的整数次幂, 这样才能在后面处理 dimension size 时, 
        将单个维度乘以或除以 2 (modify_operational_dimension 中的逻辑)
        但是例如 Eyeriss_like 的MACs就不满足以上条件(14x12), 所以最好的逻辑是：(输入的MACs, 已选择Acc的 MACs)这两个数中, 大的除以小的，是否为 2 的整数次幂
        """
        # NOTE: 这里取消了从外部更改参数 MACs 的接口，所以从 main.py 中进入的args不再有 'MACs'，但Hardware Modifier顶层依然保留这个入参，作为将来内部调试和测试
        if hasattr(args, 'MACs') and args.MACs:
            macs = int(args.MACs)
            if not (macs > 0 and (macs & (macs - 1)) == 0):  # 检查macs 是否为 2 的整数倍
                error_info['MACs'] = f'The number of MACs {macs} is not a power of 2'
        else:   # 所以，外部没有MACs参数，这里赋予它None
            args.MACs = None

        # GB size
        if hasattr(args, 'gb_size') and args.gb_size:
            gb_size = float(args.gb_size)
            if not 0.5 <= gb_size <= 10:
                error_info['global buffer size'] = f'gb_size {gb_size} MB out of range: [0.5, 10]'

        # GB Bandwidth
        if hasattr(args, 'gb_bw') and args.gb_bw:
            gb_bw = int(args.gb_bw)
            if not 64 <= gb_bw <= 1024:
                error_info['global buffer bandwidth'] = f'gb_bw {gb_bw} bit/cycle out of range: [64, 1024]'

        # Dram Bandwidth
        if hasattr(args, 'dram_bw') and args.dram_bw:
            dram_bw = int(args.dram_bw)
            if not 64 <= dram_bw <= 1024:
                error_info['dram bandwidth'] = f'dram_bw {dram_bw} bit/cycle out of range: [64, 1024]'

        # 结果返回或抛错
        if not error_info:
            return True
        else:
            print('user input arguments:\n', args)
            raise ValueError(f'Hardware Modifier Check Valid: {error_info}')

    def modify_hardware(self):
        """
        Recreate the Accelerator with config info.
        The elements required for this recreation are:
        - accelerator
            - name
            - cores
                - operational array
                    - operational unit
                    - dimension sizes
                - memory hierarchy
                    - name
                    - memory levels
                        - memory instance
                        - operands
                        - port allocation
                        - served dimensions
        """
        # Get the relevant accelerator attributes
        core = self.base_acc.get_core(1)
        memory_hierarchy = core.memory_hierarchy
        operational_array = core.operational_array

        # modify operational array
        new_operational_array = self.modify_operational_array(operational_array)

        # Initialize the new memory hierarchy
        mh_name = memory_hierarchy.name
        new_mh_name = mh_name + "_user"
        new_memory_hierarchy = MemoryHierarchy(new_operational_array, new_mh_name)

        # Add memories to the new memory hierarchy with the correct attributes
        # 这里只是把原来的mem hier 深度拷贝了一份，然后连上新的operational array，所以除了operational array其他的都没变
        for memory_level in memory_hierarchy.mem_instance_list:
            memory_instance = memory_level.memory_instance
            operands = tuple(memory_level.operands)
            port_alloc = memory_level.port_alloc_raw
            served_dimensions_vec = memory_level.served_dimensions_vec
            assert len(served_dimensions_vec) >= 1
            served_dimensions = served_dimensions_vec[0]

            if 'dram' in memory_instance.name:
                new_memory_instance = self.modify_dram(memory_instance)
            elif '_A' in memory_instance.name:  #* 使用 '_A' 字符串匹配global buffer，所以 global buffer需要注意命名规范
                new_memory_instance = self.modify_global_buffer(memory_instance)
            else:  # 非dram 和 GB，就深拷贝一份即可
                new_memory_instance = pickle_deepcopy(memory_instance)
            new_operands = pickle_deepcopy(operands)
            new_port_alloc = pickle_deepcopy(port_alloc)
            new_served_dimensions = pickle_deepcopy(served_dimensions)
            new_memory_hierarchy.add_memory(
                memory_instance=new_memory_instance,
                operands=new_operands,
                port_alloc=new_port_alloc,
                served_dimensions=new_served_dimensions,
            )

        # creat new Core
        new_core = Core(
            id=core.id,
            operational_array=new_operational_array,
            memory_hierarchy=new_memory_hierarchy,
        )

        # Create the new accelerator
        new_accelerator = Accelerator(
            name=self.acc_name,
            core_set={new_core},
        )

        return new_accelerator

    @staticmethod
    def modify_operational_dimension(dimension_sizes: list[int], nb_macs_old, nb_macs_new) -> list[int]:
        """根据给定的MACs信息配置原本的dimension sizes
        
        这里给定的macs 不一定是原本的整数倍, 而且扩大的倍数也不确定要在哪个维度上增加, 所以先在check valid的时候把不是整数倍的情况抛错, 然后在这里把最小的dim扩大, 或把最大的dim 缩小

        Args:
            dimension_sizes (list[int]): 原本的dim list, 如 [8, 32, 2, 2]
            nb_macs_old (int): 原本的MAC数, 一定是上面size中各数的乘积
            nb_macs_new (int): 用户给定的配置参数, 改变算力

        Returns:
            list[int]: 一个新的 dimension size, 数据格式相同
        """
        # 要扩大MACs
        ds = dimension_sizes.copy()
        if nb_macs_new > nb_macs_old:
            times = int(nb_macs_new / nb_macs_old)
            while times > 1:
                id = ds.index(min(ds))
                ds[id] *= 2
                times /= 2
        # 要缩小MACs
        elif nb_macs_new < nb_macs_old:
            times = int(nb_macs_old / nb_macs_new)
            while times > 1:
                id = ds.index(max(ds))
                ds[id] = int(ds[id] / 2)
                times /= 2
        else:
            raise ValueError(nb_macs_old, nb_macs_new)
        return ds

    def modify_operational_array(self, operational_array: 'OperationalArray'):
        if self.MAC_counts_str is None:
            return operational_array
        self.MAC_counts = int(self.MAC_counts_str)
        operational_unit = operational_array.unit
        dimension_sizes = operational_array.dimension_sizes
        nb_MACs = math.prod(dimension_sizes)
        new_operational_array = pickle_deepcopy(operational_array)
        if nb_MACs == self.MAC_counts:
            logger.warning(f'The number of MACs are same with selected hardware: {nb_MACs}, operational array is still {new_operational_array}')
        else:
            # Create new operational array
            new_operational_unit = pickle_deepcopy(operational_unit)
            new_dimension_sizes = self.modify_operational_dimension(dimension_sizes, nb_MACs, self.MAC_counts)  # 根据 MAC_counts 修改 operational array
            new_dimensions = {f"D{i}": new_dim_size for i, new_dim_size in enumerate(new_dimension_sizes, start=1)}
            new_operational_array = OperationalArray(new_operational_unit, new_dimensions)
            logger.info(f'operational array has been modified from {operational_array} to {new_operational_array}')
        return new_operational_array

    def modify_dram(self, dram):
        new_dram = pickle_deepcopy(dram)
        if self.ddr_bandwidth is None and self.ddr_size is None:
            return new_dram
        self.ddr_bandwidth = int(self.ddr_bandwidth)
        if dram.r_bw == self.ddr_bandwidth:
            logger.warning(f'dram bandwidth is already {self.ddr_bandwidth}')
        else:
            new_dram.r_bw = self.ddr_bandwidth
            new_dram.w_bw = self.ddr_bandwidth
            logger.info(f'ddr bandwidth is modified to {new_dram.w_bw}')
        return new_dram

    def modify_global_buffer(self, gb):
        new_gb = pickle_deepcopy(gb)
        if self.global_buffer_bandwidth_str is None and self.global_buffer_size_str is None:
            return new_gb

        logger.info(f'the original global buffer bandwidth is {gb.r_bw}, size is {gb.size/(1024*1024*8)} MB')
        if self.global_buffer_bandwidth_str is not None:
            self.gb_bw = int(self.global_buffer_bandwidth_str)
            new_gb.r_bw = self.gb_bw
            new_gb.w_bw = self.gb_bw
            logger.info(f'global buffer bandwidth is modified to {new_gb.r_bw}')
        if self.global_buffer_size_str is not None:
            new_gb.size = float(self.global_buffer_size_str) * 1024 * 1024 * 8  # MB to bit
            new_gb.name = f'sram_{self.global_buffer_size_str}MB_A'
            logger.info(f'global buffer size is modified to {new_gb.size/(1024*1024*8)} MB')

        new_gb.r_cost = CoreGenerator._estimate_port_cost(True, new_gb.size, new_gb.r_bw)
        new_gb.w_cost = CoreGenerator._estimate_port_cost(False, new_gb.size, new_gb.w_bw)
        logger.info(f'global buffer port energy has been modified to {new_gb.r_cost} and {new_gb.w_cost}')
        return new_gb


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Setup npuperf inputs")
    parser.add_argument('--nn', metavar='Network name', required=True, help='module name to hhb networks, e.g. fsrcnn2x')
    parser.add_argument('--hw', metavar='Hardware name', required=True, help='module name to the accelerator, e.g. example_wioGB')
    parser.add_argument('--MACs',
                        metavar='hardware config info: the numbers of MACs',
                        required=False,
                        help='Optional: change number of MACs based on selected hardware, e.g. 2048')
    parser.add_argument('--gb_size',
                        metavar='hardware config info: the global buffer size (MB)',
                        required=False,
                        help='Optional: change Global Buffer size based on selected hardware, e.g. 3')
    parser.add_argument('--gb_bw',
                        metavar='hardware config info: the global buffer bandwidth (bit/cycle)',
                        required=False,
                        help='Optional: change Global Buffer bandwidth based on selected hardware, e.g. 256')
    parser.add_argument('--dram_bw',
                        metavar='hardware config info: the dram bandwidth (bit/cycle)',
                        required=False,
                        help='Optional: change dram bandwidth based on selected hardware, e.g. 256')

    args = parser.parse_args()
    from npuperf.inputs.HW.Meta_prototype import accelerator
    modi = HardwareModifier.from_arg(acc=accelerator, config_info=args)
    acc_modified = modi.get_accelerator()
    # old acc
    print('---' * 30)
    print(f'old acc name: {accelerator.name}')
    for k, v in accelerator.get_core(1).__jsonrepr__().items():
        print(k, v)
    # new
    print('---' * 30)
    print(f'new acc name: {acc_modified.name}')
    for k, v in acc_modified.get_core(1).__jsonrepr__().items():
        print(k, v)
