import logging

import numpy as np

from npuperf.classes.mapping.spatial.spatial_mapping import SpatialMapping
from npuperf.classes.stages.Stage import Stage

logger = logging.getLogger(__name__)


class SpatialMappingConversionStage(Stage):
    """
    Pipeline stage that converts the spatial mapping from a
    user-provided spatial mapping across operational array dimensions
    to the internal spatial mapping representation used in the cost model.
    """

    def __init__(self, list_of_callables, *, accelerator, layer, **kwargs):
        """
        Initialize the accelerator and layer attributes.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.check_layer(layer)  # raise ValueError in case anything is wrong
        self.layer = layer
        self.accelerator = accelerator

    @staticmethod
    def check_layer(layer):
        """
        Check the layer attribute of the main_inputs:
        check that the layer includes:
        - the core which it is allocated to
        - the user-defined spatial mapping
        If not, a ValueError is raised.
        :return: True
        """
        if not layer.core_allocation:
            logger.critical(f"Layer {layer} has no core allocation.")
            raise ValueError()
        if not layer.user_spatial_mapping:
            logger.critical(f"Layer {layer} has no user-defined spatial mapping.")
            raise ValueError("Missing spatial mapping for spatial mapping conversion")

        return True

    def run(self):

        user_spatial_mapping = self.layer.user_spatial_mapping.copy()  # 这里需要 copy，否则对当前层的处理可能影响到整个网络
        spatial_mapping = self.convert_user_spatial_mapping(user_spatial_mapping)
        kwargs = self.kwargs.copy()
        kwargs['spatial_mapping'] = spatial_mapping
        kwargs['accelerator'] = self.accelerator
        kwargs['layer'] = self.layer

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def convert_user_spatial_mapping(self, user_spatial_mapping):
        """
        Convert the user-defined spatial mapping across operational array dimensions
        to the internal SpatialMapping representation.
        For this conversion we need to know:
        - the user defined spatial mapping
        - the core (i.e. operational array) on which the unrolling happens,
          and the memory hierarchy that is connected to that operational array.
        :param user_spatial_mapping: The user-defined spatial mapping to be converted.
        Returns: A SpatialMapping object with the converted spatial mapping.
        """
        # Adjust the user defined spatial mapping size based on the operational array dimension and the layer dimension:
        # E.g. user-provided unrolling is 16 but operational array dimension size is only 12: change unrolling to 12
        # E.g. user-provided unrolling is 16 but layer dimension is only 12: change unrolling to 12
        # E.g. user-provided unrolling is 16 but layer dimension is not a multiple of 16: change unrolling to fractional number
        # so that the temporal remainder is an integer.
        # 第三点：例如，用户提供的展开为16，但layer的维度不是16的倍数：将展开更改为小数，使得时间余数是整数

        core_id = self.layer.core_allocation
        core = self.accelerator.get_core(core_id)
        mem_hierarchy = core.memory_hierarchy
        oa_dims = core.operational_array.dimensions
        layer_dim_sizes = self.layer.loop_dim_size
        for spatial_dim_name, spatial_loop in user_spatial_mapping.items():
            # Check 1: Limit unrolling if operational array dimension is smaller than provided unrolling
            oa_dim_size = next((oa_dim for oa_dim in oa_dims if oa_dim.name == spatial_dim_name)).size
            (loop_dim_unrolled, loop_size_unrolled) = spatial_loop
            loop_size_unrolled = min(oa_dim_size, loop_size_unrolled)
            # Check 2: Limit unrolling if layer dimension is smaller than provided unrolling
            layer_dim_size = layer_dim_sizes[loop_dim_unrolled]
            loop_size_unrolled = min(layer_dim_size, loop_size_unrolled)
            # Check 3: Adjust unrolling if it is not a multiple of the layer dimension size
            temporal_remainder = int(np.ceil(layer_dim_size / loop_size_unrolled))
            loop_size_unrolled = layer_dim_size / temporal_remainder
            # Set the adjusted unrolling size in the original user_spatial_mapping dict
            user_spatial_mapping[spatial_dim_name] = (loop_dim_unrolled, loop_size_unrolled)

        spatial_mapping_dict = {}
        layer_to_mem_op = self.layer.memory_operand_links
        mem_to_layer_op = {mem_op: layer_op for (layer_op, mem_op) in layer_to_mem_op.items()}
        core_id = self.layer.core_allocation
        mem_hierarchy = self.accelerator.get_core(core_id).memory_hierarchy
        # 其实这个复杂的循环是为了将用户定义的空间映射，从下向上铺在内存层级上，必须注意第一项是MAC阵列的mapping维度，后面才是reg或生sram的空间mapping，完全没考虑容量问题，只考虑服务维度问题
        for mem_op, layer_op in mem_to_layer_op.items():  #mem_hierarchy.operands:
            # 遍历一个layer中需要的三个操作数，I1, I2, O
            user_sm_copy = user_spatial_mapping.copy()
            # layer_op = mem_to_layer_op[mem_op]
            spatial_mapping_dict[layer_op] = []
            memory_levels = mem_hierarchy.get_memory_levels(mem_op, ) #拿到这个mem_op的全部内存level
            # 遍历当前操作数所在的每一个memory level
            for memory_level in memory_levels:
                spatial_mapping_lvl = []
                served_dimensions = memory_level.served_dimensions
                for dimension in served_dimensions:
                    #遍历当前memory level的每一个服务维度，看是否在用户定义的空间映射中
                    dim_name = dimension.name
                    if dim_name in user_sm_copy:
                        # The dimension name is present in the user defined spatial mapping
                        # Add the spatial loop of this dimension to the spatial mapping
                        spatial_loop = user_sm_copy[dim_name]
                        spatial_mapping_lvl.append(spatial_loop)
                        # Then remove this dim_name and spatial loop key value pair from the dict
                        # as the spatial mapping representation is a level-by-level one.
                        del user_sm_copy[dim_name]
                spatial_mapping_dict[layer_op].append(spatial_mapping_lvl)

            # After we have gone through the memory levels, if there are still user-defined dimensions
            # present, add them as the top level. Otherwise add an empty list to make arch levels correct:
            # because first list we added was the operational array level.
            # 如果用户定义的空间映射中还有剩余的维度没有被添加到空间映射中，直接加在顶层，一般都是没有，直接加一个空[]
            top_level_spatial_mapping = [spatial_loop for (dim_name, spatial_loop) in user_sm_copy.items()]
            spatial_mapping_dict[layer_op].append(top_level_spatial_mapping)

        return SpatialMapping(spatial_mapping_dict=spatial_mapping_dict, layer_node=self.layer)


# 以 Meta_prototype_DF 为例，某一层的 spatial_mapping_dict 是以下这样，O和W 有5层，I有4层。必须比mem hier 中多一层，因为最低是从PE array开始算的
# {'O': [[('C', 2.0)], [('K', 32.0), ('OY', 4.0), ('OX', 4.0)], [], [], []],
#  'W': [[('OY', 4.0), ('OX', 4.0)], [('K', 32.0), ('C', 2.0)], [], [], []],
#  'I': [[('K', 32.0), ('OY', 4.0), ('OX', 4.0), ('C', 2.0)], [], [], []]}
if __name__ == '__main__':

    class Dummy(Stage):

        def is_leaf(self):
            return True

        def run(self):
            yield None, self.kwargs

    from npuperf.classes.stages.Stage import MainStage
    from npuperf.classes.workload.layer_node import LayerNode
    from npuperf.inputs.HW.example_wioGB import accelerator
    from npuperf.inputs.WL_fromjson.Meta_prototype.workload_mv1 import workload

    layer = workload[0]
    layer_node = LayerNode(0, layer)
    DUT = MainStage(list_of_callables=[SpatialMappingConversionStage, Dummy], accelerator=accelerator, layer=layer_node)
    for i in DUT.run():
        print(i)
# result: 下面是 spatial_mapping_dict，是SpatialMapping类的入参
# {'O': [[('C', 32.0)], [('OY', 2.0), ('OX', 2.0), ('K', 8.0)], [], []], 
#  'W': [[('OY', 2.0), ('OX', 2.0)], [('C', 32.0), ('K', 8.0)], [], []], 
#  'I': [[('OY', 2.0), ('OX', 2.0), ('C', 32.0), ('K', 8.0)], [], []]}