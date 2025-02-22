import json

from npuperf.classes.hardware.architecture.accelerator import Accelerator
from npuperf.classes.opt.hw_gen.core_generator import CoreGenerator


class HardwareGenerator:
    """use get_accelerator() method to return a accelerator
    """

    def __init__(self, json_HW: str | dict):
        """
        Args:
            json_HW (str): path to json format json path
            - e.g.: json_HW = 'npuperf/inputs/hw_config/example_wioGB.json'
        """
        self.parse_json_hw(json_HW)
        self.core_set()

    def get_accelerator(self):
        return self.accelerator

    def parse_json_hw(self, json_HW: str | dict):
        if isinstance(json_HW, dict):
            assert json_HW.get('name', None), 'Your json format hardware config file has no "name" item, please add one'
            self.hw_name = json_HW['name']  # 如果传入的是dict，那么从dict中的'name'字段取出self.name，作为accelerator的name。所以，若给定dict，name字段是必须的
        elif isinstance(json_HW, str):
            self.hw_name = json_HW.split('/')[-1][:-5]  # 如果传入的是路径，则取出路径中的json文件名，作为hardware name。所以，若给定path，name字段是无用的
            with open(json_HW) as f:
                json_HW = json.load(f)
        core_info = self.change_input_format(json_HW)
        self.core_info = core_info

    def change_input_format(self, input_dict):
        """ json format hardware config file is like:
        {
            "MAC_unroll": {
                "K": 8,
                "C": 32,
                "OX": 4,
                "OY": 2
            },
            "local_buffers": [
                {
                    "op": "W",
                    "size": 128
                },
                {
                    "op": "I",
                    "size": 64
                },
                {
                    "op": "O",
                    "size": 128
                }
            ],
            "global_buffer": {
                "op": "W/I/O",
                "size": 1,
                "bandwidth": 256
            },
            "dram": {
                "op": "W/I/O",
                "bandwidth": 192
            }
        }
        """
        mac_unroll = input_dict.get("MAC_unroll", {})
        output_dict = {
            "MAC_unroll": {
                f"D{i}": (k, v)
                for i, (k, v) in enumerate(mac_unroll.items(), start=1)
            },
            "local_buffers": [{
                "op": buffer["op"],
                "size": buffer["size"] * 1024 * 8
            } for buffer in input_dict.get("local_buffers", [])],
            "global_buffer": {
                "op": input_dict.get("global_buffer", {}).get("op", ""),
                "size": input_dict.get("global_buffer", {}).get("size", 0) * 1024 * 1024 * 8,
                "bandwidth": input_dict.get("global_buffer", {}).get("bandwidth", 0)
            } if input_dict.get("global_buffer") else None,
            "dram": {
                "op": input_dict.get("dram", {}).get("op", ""),
                "size": input_dict.get("dram", {}).get("size", 0) * 1024 * 1024 * 1024 * 8,
                "bandwidth": input_dict.get("dram", {}).get("bandwidth", 0)
            } if input_dict.get("dram") else None,
        }
        return output_dict

    def core_set(self):
        core_gen = CoreGenerator.from_dict(self.core_info)
        self.core = core_gen.get_core()
        cores = {self.core}
        self.accelerator = Accelerator(self.hw_name, cores)


if __name__ == '__main__':
    from npuperf.inputs.HW.Meta_prototype_DF import accelerator  # reference for test
    json_HW = 'npuperf/inputs/hw_config/Meta_prototype_DF.json'  # test this
    Hardware_gen = HardwareGenerator(json_HW)
    accelerator_gen = Hardware_gen.get_accelerator()
    print(Hardware_gen.core_info)
    # for k,v in Hardware_gen.core_info.items():
    #     print(k,v)

    # form HW.*.py
    print('---' * 30)
    print(f'from .py acc name: {accelerator.name}')
    for k, v in accelerator.get_core(1).__jsonrepr__().items():
        print(k, v)

    # form hw gen
    print('---' * 30)
    print(f'from hw gen acc name: {accelerator_gen.name}')
    for k, v in accelerator_gen.get_core(1).__jsonrepr__().items():
        print(k, v)

    ############ test dict input ##########
    json_HW = {
        "name": 'hahaha',
        "MAC_unroll": {
            "K": 32,
            "C": 2,
            "OX": 4,
            "OY": 4
        },
        "local_buffers": [{
            "op": "W",
            "size": 32
        }, {
            "op": "W",
            "size": 1024
        }, {
            "op": "I/O",
            "size": 64
        }],
        "global_buffer": {
            "op": "O/I",
            "size": 1,
            "bandwidth": 1024
        },
        "dram": {
            "op": "W/I/O",
            "bandwidth": 64
        }
    }
    Hardware_gen = HardwareGenerator(json_HW)
    accelerator_gen = Hardware_gen.get_accelerator()
    # form hw gen with dict input
    print('---' * 30)
    print(f'from hw gen acc name: {accelerator_gen.name}')
    for k, v in accelerator_gen.get_core(1).__jsonrepr__().items():
        print(k, v)
