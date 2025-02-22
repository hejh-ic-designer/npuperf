import importlib
import os
from pprint import pprint
from typing import Dict, Any

"""
本文件用于将已有workload 文件中的spatial mapping部分改成适用于其他硬件, 输入一个workload 和一个现有架构的spatial mapping, 就可以输出适用于现有架构的workload
"""

class Change_spm_for_workload:
    """ old workload.py + mapping.py --> new workload.py """
    def __init__(self, workload_path, mapping_path, export_path):
        self.workload: Dict[int, Dict[str, Any]] = self.load_module(workload_path, "workload")
        self.mapping: Dict[str, Dict] = self.load_module(mapping_path, "mapping")
        self.export_path = export_path

    def load_module(self, module_path, module_name):
        try:
            module = importlib.import_module(module_path)
            return getattr(module, module_name)     # 从module 中取出module_name, 相当于 module.workload, module.mapping
        except ModuleNotFoundError:
            raise ImportError(f"Module '{module_path}' not found")

    def run(self):
        for layer in self.workload.values():
            if layer['equation'] == 'input':
                continue
            try:
                spatial_mapping = self.mapping[layer['operator_type']]['spatial_mapping']   # 替换 spm
                layer['spatial_mapping'] = spatial_mapping
            except:
                print(f'Warning: "{layer["operator_type"]}" operator not found in mapping file')

        self.export()

    def export(self):
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)

        with open(self.export_path, 'w') as ff:
            print('workload =', file=ff, end='')
            pprint(object=self.workload, stream=ff, indent=4, width=150, sort_dicts=False)

if __name__ == '__main__':
    workload_name = 'workload_resnet18'
    DLA = 'example_wioGB'

    # workload_path = f'inputs.WL_fromjson.Tesla_NPU_like.{workload_name}'
    workload_path = f'npuperf.inputs.WL.Meta_prototype.{workload_name}'
    mapping_path = f'npuperf.inputs.Mapping.{DLA}'
    export_path = f"../npuperf/inputs/WL_fromjson/{DLA}/{workload_name}.py"

    change_sp = Change_spm_for_workload(workload_path, mapping_path, export_path)
    change_sp.run()
    print(f'DONE, export file at {export_path}')
