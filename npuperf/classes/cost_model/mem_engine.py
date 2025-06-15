import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

from npuperf.classes.depthfirst.data_copy_layer import (DataCopyAction, DataCopyLayer)
from npuperf.classes.workload.layer_node import LayerNode
from npuperf.classes.workload.mem_node import MemNode
from npuperf.utils import pickle_deepcopy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from npuperf.classes.hardware.architecture.accelerator import Accelerator


class MemoryOperatorEvaluation:
    """和memory相关的算子评估模型
    """

    def __init__(self, *, accelerator: 'Accelerator', layer: LayerNode | MemNode) -> None:
        self.accelerator = accelerator
        self.layer = layer
        self.core_id = layer.core_allocation
        self.mem_hierarchy_dict = accelerator.get_core(self.core_id).get_memory_hierarchy_dict()
        self.run()

    def run(self):
        self.calc_extra_datacopy()

    def calc_extra_datacopy(self):
        if isinstance(self.layer, LayerNode):
            if self.layer.post_process is not None:  # 这里是对于workload 中 post process 字段的处理
                #! 这是原本对于pixel shuffle的建模，但是考虑到数据往外传递的过程和普通卷积无法区分，所以这部分先不采用，有待商榷
                pass
                #todo data source 和 data destination 的表达要优化，现在只是考虑了从 DRAM 到次最高级的mem lv搬数据，下面是手捏的，代码不好维护
                # d_des = self.mem_hierarchy_dict['I1'][-1]
                # d_des_str = 'I1'
                # d_des_int = len(self.mem_hierarchy_dict['I1']) -1   # 应该是从0开始表示的，所以取出来要减去 1
                # d_des = (d_des_str, d_des_int)      # data source 是 I1 的最高级别内存，肯定是DRAM了

                # d_src = self.mem_hierarchy_dict['O'][-2]
                # d_src_str = 'O'
                # d_src_int = len(self.mem_hierarchy_dict['O']) -2    # 次最高级，那就减去 2
                # d_src = (d_src_str, d_src_int)      # data source 是 O 的次最高级别内存，一般是DRAM 下面的一级

                # # logger.info(f"Post_process of {self.layer.post_process} executing, data amount is {self.layer.operand_size_bit['O']}, layer : {self.layer.TYPE}")
                # # 1 个 data copy action, 从 片上搬到 DRAM
                # dca1 = DataCopyAction(data_amount = self.layer.operand_size_bit['O'], data_source = d_src, data_destination = d_des, core = self.accelerator.get_core(self.core_id))
                # dcl = DataCopyLayer(layer_id= self.layer.id, data_copy_actions= [dca1], accelerator= self.accelerator, core_id= self.core_id)
                # self.extra_data_copy_en = dcl.energy_total
                # self.extra_data_copy_la = dcl.latency_total2
            else:
                self.extra_data_copy_en = 0
                self.extra_data_copy_la = 0

        elif isinstance(self.layer, MemNode):
            self.energy_total = 0
            self.latency_total2 = 0
            # 从 DRAM 到次最高级别的内存搬数据
            # 这里的 data source 和 data destination 是手动指定的，代码中是从 DRAM 搬到 O 的次最高级别内存
            # 然后再从 O 的次最高级别内存搬回DRAM

            d_src_mem = self.mem_hierarchy_dict['I1'][-1]
            d_src_str = 'I1'
            d_src_int = len(self.mem_hierarchy_dict['I1']) - 1  # 应该是从0开始表示的，所以取出来要减去 1
            d_src = (d_src_str, d_src_int)  # data source 是 I1 的最高级别内存，肯定是DRAM了

            d_des_mem = self.mem_hierarchy_dict['O'][-2]
            d_des_str = 'O'
            d_des_int = len(self.mem_hierarchy_dict['O']) - 2  # 次最高级，那就减去 2
            d_des = (d_des_str, d_des_int)  # data source 是 O 的次最高级别内存，一般是DRAM 下面的一级

            logger.info(
                f"Processing {self.layer.equation} Layer, data amount is {self.layer.operand_size_bit['O']}, data src is {d_src_mem}, and des is {d_des_mem}"
            )
            # 两个 data copy action, 从 DRAM搬进来，再搬出去
            dca1 = DataCopyAction(data_amount=self.layer.operand_size_bit['O'],
                                  data_source=d_src,
                                  data_destination=d_des,
                                  core=self.accelerator.get_core(self.core_id))
            dca2 = DataCopyAction(data_amount=self.layer.operand_size_bit['O'],
                                  data_source=d_des,
                                  data_destination=d_src,
                                  core=self.accelerator.get_core(self.core_id))
            dcl = DataCopyLayer(layer_id=self.layer.id, data_copy_actions=[dca1, dca2], accelerator=self.accelerator, core_id=self.core_id)
            self.extra_data_copy_en = dcl.energy_total
            self.extra_data_copy_la = dcl.latency_total2

        else:
            raise TypeError("the layer is not type of LayerNode or MemNode.")

    def __simplejsonrepr__(self):
        """
        Simple JSON representation used for saving this object to a simple json file.
        """
        return {
            "layer name": self.layer.name,
            "layer id": self.layer.id,
            "energy(mJ)": (self.extra_data_copy_en) / 1e9,
            "latency(mC)": (self.extra_data_copy_la) / 1e6,
        }

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a complete json file.
        """
        return {
            "layer name": self.layer.name,
            "layer id": self.layer.id,
            "energy(mJ)": (self.extra_data_copy_en) / 1e9,
            "latency(mC)": (self.extra_data_copy_la) / 1e6,
        }

    def __str__(self):
        return f"MemoryOperatorEvaluation(layer={self.layer})"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        sum = pickle_deepcopy(self)

        sum.extra_data_copy_en += other.extra_data_copy_en
        sum.extra_data_copy_la += other.extra_data_copy_la
        return sum
