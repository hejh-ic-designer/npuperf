import logging
import os
from pprint import pprint
from typing import TYPE_CHECKING, Any, Dict

from npuperf.classes.stages.Stage import Stage

if TYPE_CHECKING:
    from npuperf.classes.hardware.architecture.accelerator import Accelerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MappingGeneratorStage(Stage):
    """用于产生Mapping dict, 在 JsonWorkloadParser中解析到Workload 中

    仅考虑 1 Core
    """

    def __init__(self, list_of_callables, *, accelerator: 'Accelerator', **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        dim = accelerator.get_core(1).operational_array.dimensions
        # 类似 {'D1': 8, 'D2': 32, 'D3': 2, 'D4': 2}
        self.dim = {f'D{dimension.id + 1}': dimension.size for dimension in dim}
        self.nb_dim = accelerator.get_core(1).operational_array.nb_dimensions
        assert self.nb_dim in [2, 3, 4], f'number of dimension: {self.nb_dim} is out of range [2, 3, 4] !'
        self.DLA_name = accelerator.name

    def run(self):
        mapping_dict = self.gen_mapping()
        logger.debug(f'Generate Mapping Dict for {self.DLA_name} DONE! spatial mapping for Conv is {mapping_dict["Conv"]["spatial_mapping"]}')
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:],
                                              json_mapping_path_or_dict=mapping_dict,
                                              accelerator=self.accelerator,
                                              **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def gen_mapping(self, export_to_file=False) -> Dict[str, Dict[str, Any]]:  #* Here to change do/don't export mapping to file
        mapping_dict = {
            'DLA_name': self.DLA_name,
            'partial_sum_precision': self.parse_O_reg_bw_from_DLA(self.accelerator),
        }
        for op in self.check_op():
            mapping_op = {
                'core_allocation': 1,
                'spatial_mapping': self.set_spatial_mapping(op),
                'memory_operand_links': self.set_memory_operand_links(op),
            }
            mapping_dict[op] = mapping_op
        self.mapping_dict = mapping_dict
        if export_to_file:
            self.export()
        return self.mapping_dict

    def parse_O_reg_bw_from_DLA(self, DLA: 'Accelerator') -> int:
        """给定 accelerator object, 解析出 O_reg rd/wr bandwidth, 此值将用于 partial sum precision

        Returns:
            int: partial sum precision, equal to O reg file Bandwidth (16, 24, 32)
        """
        O_reg_bw = DLA.get_core(1).get_memory_level(mem_op='O', mem_lv=0).memory_instance.w_bw
        assert O_reg_bw in [16, 24, 32], f'partial sum precision parse ERROR, not in [16, 24, 32], it is {O_reg_bw}'
        logger.debug(f'Set partial sum precision: {O_reg_bw} bit')
        return O_reg_bw

    def check_op(self) -> list[str]:
        """这里的算子应该是npuperf能支持的所有的算子类型, 不仅仅是要执行的网络包含的所有 op。
        这里的 op 应该不包含mem op, 因为mem op在parser 里作了 core_allocation 和 memory operand links
        """
        self.op_OWI = ['Conv', 'dw_Conv', 'Pool', 'Dense']
        self.op_OWI_and_OXY = ['Add', 'Subtract', 'Mul', 'Matmul']
        self.op_other = ['Input']
        op_list = self.op_OWI + self.op_OWI_and_OXY + self.op_other
        return op_list

    def set_spatial_mapping(self, op) -> Dict[str, tuple]:
        """返回值类似 {"D1": ("K", 8), "D2": ("C", 32), "D3": ("OX", 2), "D4": ("OY", 2)}

        首先, self.nb_dim 是MAC unroll 维度的数量, 可能为 2-4, 对每种情况需要设定要unroll 的维度
        unroll 2 dims: K, C
        unroll 3 dims: K, OX, OY
        unroll 4 dims: K, C, OX, OY
        设定spatial mapping的时候, 先给Conv设定, 然后由Conv的推导其他算子的
        """
        if op == 'Input':
            return None

        def case_conv():
            if self.nb_dim == 4:
                unroll_dim = ['K', 'C', 'OX', 'OY']
            elif self.nb_dim == 3:
                unroll_dim = ['K', 'OX', 'OY']
            else:  # self.nb_dim == 2
                unroll_dim = ['K', 'C']

            return {dim_id: (unroll_dim[id], size) for id, (dim_id, size) in enumerate(self.dim.items())}

        def case_dense():
            if self.nb_dim == 4:
                unroll_dim = ['K', 'C', 'B', 'B']
            elif self.nb_dim == 3:
                unroll_dim = ['K', 'B', 'B']
            else:  # self.nb_dim == 2
                unroll_dim = ['K', 'C']

            return {dim_id: (unroll_dim[id], size) for id, (dim_id, size) in enumerate(self.dim.items())}

        def case_dwconv():
            ''' 虽然 dw_conv 也是 g_unroll, 但 OX 和 OY 维度的unroll是需要设置的, 所以单独拿出来 '''
            if self.nb_dim == 4:
                unroll_dim = ['G', 'OX', 'OY']
                dim = self.dim.copy()
                dim.pop('D2')  # copy self.dim 并删掉第二个 key
            elif self.nb_dim == 3:
                unroll_dim = ['G', 'OX', 'OY']
                dim = self.dim.copy()
            else:  # self.nb_dim == 2:
                unroll_dim = ['G']
                # 只取 self.dim 的第一个出来
                dim = dict(list(self.dim.items())[:1])
            return {dim_id: (unroll_dim[id], size) for id, (dim_id, size) in enumerate(dim.items())}

        def case_matmul():
            if self.nb_dim == 4:
                unroll_dim = ['K', 'C', 'OX', 'OX']
            elif self.nb_dim == 3:
                unroll_dim = ['K', 'C', 'OX']
            else:  # self.nb_dim == 2
                unroll_dim = ['K', 'C']

            return {dim_id: (unroll_dim[id], size) for id, (dim_id, size) in enumerate(self.dim.items())}

        def case_other():
            ''' 这里设置的是 Add, Mul, Subtract, Pool 算子, 这三个算子的空间映射只有 G 的维度, 不应该有OX, OY 的维度 '''
            unroll_dim = ['G']
            dim = dict(list(self.dim.items())[:1])  # 只取 self.dim 的第一个出来
            return {dim_id: (unroll_dim[id], size) for id, (dim_id, size) in enumerate(dim.items())}

        cases = {
            'Conv': case_conv,
            'Dense': case_dense,
            'dw_Conv': case_dwconv,
            'Matmul': case_matmul,
        }
        return cases.get(op, case_other)()

    def set_memory_operand_links(self, op) -> dict:
        if op in self.op_OWI:
            return {"O": "O", "W": "I2", "I": "I1"}
        elif op in self.op_OWI_and_OXY:
            return {'OWI': {"O": "O", "W": "I2", "I": "I1"}, 'OXY': {"O": "O", "X": "I1", "Y": "I1"}}
        elif op in self.op_other:
            return {"O": "I1"}  # Input 第一层
        else:
            raise KeyError(f'unknown operator.')

    def export(self):
        """export .py format mapping dict to file at export_path
        """
        export_path = f"npuperf/inputs/Mapping/{self.DLA_name}_gen.py"
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        with open(export_path, 'w') as ff:
            print('mapping =', file=ff, end='')
            pprint(object=self.mapping_dict, stream=ff, indent=4, width=90, sort_dicts=False)
        logger.info(f'mapping File export DONE!, at path {export_path} \n' + '---' * 60)


if __name__ == '__main__':

    class Dummy(Stage):

        def is_leaf(self):
            return True

        def run(self):
            yield None, self.kwargs

    from pprint import pprint
    from npuperf.classes.stages.Stage import MainStage
    # from npuperf.inputs.HW.Eyeriss_like import accelerator
    # from npuperf.inputs.HW.TPU_like import accelerator
    # from npuperf.inputs.HW.Tesla_NPU_like import accelerator
    from npuperf.inputs.HW.example_wioGB import accelerator
    DUT = MainStage([MappingGeneratorStage, Dummy], accelerator=accelerator)
    for l in DUT.run():
        pprint(l, width=160)
