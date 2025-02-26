import logging
from math import ceil

from npuperf.classes.hardware.architecture.accelerator import Accelerator
from npuperf.classes.hardware.architecture.core import Core
from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from npuperf.classes.hardware.architecture.operational_array import OperationalArray
from npuperf.classes.stages.Stage import Stage
from npuperf.utils import pickle_deepcopy

logger = logging.getLogger(__name__)


## This stage scales the PE array of the given accelerator.
## Because the user-defined spatial mapping resides in the different workload layer nodes,
## We also have to modify those to scale accordingly
class PEArrayScalingStage(Stage):
    """设置一个缩放系数 pe_array_scaling, 把MAC_unroll的每个维度乘以系数, 再把workload的spatial mapping乘以系数
    
    并不会对 mem hier 产生影响
    """

    def __init__(self, list_of_callables, *, workload, accelerator, pe_array_scaling, **kwargs):
        super().__init__(list_of_callables, **kwargs)

        ## SANITY CHECKS
        # Only allow scaling factors that are a power of 2
        assert pe_array_scaling in [2**i for i in range(-3, 3)]  # 这里是从 1/8 倍到 4 倍
        # Make sure there's only one core so that the correct one is scaled
        # If your accelerator has more cores, modify the function below
        assert len(accelerator.cores) == 1

        self.workload = workload
        self.accelerator = accelerator
        self.pe_array_scaling = pe_array_scaling

    def run(self):
        scaled_accelerator = self.generate_scaled_accelerator()
        modified_workload = self.scale_workload_spatial_mapping()
        sub_stage = self.list_of_callables[0]( self.list_of_callables[1:], workload=modified_workload, accelerator=scaled_accelerator, **self.kwargs, )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def generate_scaled_accelerator(self):
        """
        Recreate the Accelerator with PE array dimension scaling in all dimensions.
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
        core = next(iter(self.accelerator.cores))
        operational_array = core.operational_array
        operational_unit = operational_array.unit
        dimension_sizes = operational_array.dimension_sizes
        memory_hierarchy = core.memory_hierarchy

        # Create new operational array
        new_operational_unit = pickle_deepcopy(operational_unit)
        new_dimension_sizes = [ceil(self.pe_array_scaling * dim_size) for dim_size in dimension_sizes]  # 把 dim size 统一的缩放一定的倍数
        new_dimensions = {f"D{i}": new_dim_size for i, new_dim_size in enumerate(new_dimension_sizes, start=1)}
        new_operational_array = OperationalArray(new_operational_unit, new_dimensions)

        # Initialize the new memory hierarchy
        mh_name = memory_hierarchy.name
        new_mh_name = mh_name + "-scaled"
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

        # Create the new core
        id = core.id
        dataflows = core.dataflows
        if dataflows is not None:
            raise NotImplementedError("Scale your core-defined dataflows accordingly here.")

        new_id = id
        new_dataflows = pickle_deepcopy(dataflows)
        new_core = Core(
            id=new_id,
            operational_array=new_operational_array,
            memory_hierarchy=new_memory_hierarchy,
            dataflows=new_dataflows,
        )

        # Create the new accelerator
        name = self.accelerator.name
        new_name = name + "-scaled"
        new_cores = {new_core}
        new_accelerator = Accelerator(
            name=new_name,
            core_set=new_cores,
        )

        return new_accelerator

    def scale_workload_spatial_mapping(self):
        """
        Scale the user-defined mappings for each layer.
        """
        modified_workload = pickle_deepcopy(self.workload)
        for node in modified_workload.nodes():
            if hasattr(node, "user_spatial_mapping") and node.user_spatial_mapping:
                for array_dim, (layer_dim, size) in node.user_spatial_mapping.items():
                    if size != 1:   # 如果 size 本来就是 1，则不用变，因为为 1 的话已经不会有 spatial unroll了
                        node.user_spatial_mapping[array_dim] = (layer_dim, self.pe_array_scaling * size,)
        return modified_workload


if __name__ == '__main__':

    class Dummy(Stage):

        def is_leaf(self):
            return True

        def run(self):
            yield None, self.kwargs

    from npuperf.classes.stages.Stage import MainStage
    from npuperf.inputs.HW.example_wioGB import accelerator
    from npuperf.inputs.WL_fromjson.Meta_prototype.workload_mv1 import workload
    from npuperf.classes.workload.dnn_workload import DNNWorkload
    workload = DNNWorkload(workload)
    DUT = MainStage([PEArrayScalingStage, Dummy], accelerator=accelerator, pe_array_scaling=2, workload=workload)
    for l in DUT.run():
        print(l)
        _, kwg = l
        core = kwg['accelerator'].get_core(1)
        cfg = core.__jsonrepr__()
        for k, v in cfg.items():
            print(k, v)
