# Case study 2 的架构生成脚本
# 架构生成中：
# 变的：算力，LB大小配比
# 不变的：拓扑结构，DRAM配置
import math

drams = [
    {
        'op': "W/I/O",
        'bandwidth': 4096,
        'size': 24 * 1024 * 1024 * 1024 * 8,
    },
]


# K, C, OX 对应于 N, K, M
MAC_unrolls = {
    2048:  {"K": 8 , "C": 32, "OX":8 },
    4096:  {"K": 8 , "C": 32, "OX":16},
    8192:  {"K": 16, "C": 32, "OX":16},
    16384: {"K": 16, "C": 64, "OX":16},
    32768: {"K": 16, "C": 64, "OX":32},
    65536: {"K": 32, "C": 64, "OX":32},
    131072: {"K": 32, "C": 64, "OX":64},
    262144: {"K": 64, "C": 64, "OX":64},
    524288: {"K": 64, "C": 128, "OX":64},
    1048576: {"K": 64, "C": 128, "OX":128},
}

local_buffers = {
    2048: 256,
    4096: 512,
    8192: 1024,
    16384: 2048,
    32768: 4096,
    65536: 8192,
    131072: 8192,
    262144: 8192,
    524288: 8192,
    1048576: 8192,
}


def get_buffer_size(total_size, ratio):
    """计算buffer的size

    Args:
        total_size (int): 总的local buffer大小, 单位为KB
        ratio (float): buffer_1 和 buffer_2 大小的比例, 如果是 1:2 就是 0.5, 如果是 1:1 就是 1

    Returns:
        tuple: 两个buffer的大小
    """
    a = math.ceil(total_size * (ratio / (ratio + 1)))
    b = total_size - a
    return (a, b)


class CASE_gen:

    def __init__(self, ratio) -> None:
        self.hw_topology = {
            "local_buffers": [{
                "op": "I/W",
            }, {
                "op": "O",
            }],
            "dram": drams[0]
        }
        self.ratio = ratio

    def gen_cases(self):
        for nb_MAC, MAC_unroll in MAC_unrolls.items():
            IW_lb_size, O_lb_size = get_buffer_size(local_buffers[nb_MAC], self.ratio)
            new_hw = self.hw_topology.copy()
            new_hw["name"] = f'DLA_{nb_MAC}'
            new_hw['MAC_unroll'] = MAC_unroll
            new_hw['local_buffers'][0]["size"] = IW_lb_size
            new_hw['local_buffers'][1]["size"] = O_lb_size
            yield new_hw


if __name__ == '__main__':
    from npuperf.classes.opt.hw_gen.hardware_generator import HardwareGenerator
    i = 0
    for MEM_hier in CASE_gen(ratio=0.5).gen_cases():
        i += 1
        new_DLA = HardwareGenerator(MEM_hier).get_accelerator()
        print(f'id: {i} @{new_DLA.name}: {MEM_hier}')
        print('check IW_LB port list: ', new_DLA.get_core(1).get_memory_level('I2', 1).get_port_list())
        for k, v in new_DLA.get_core(1).__jsonrepr__().items():
            print(k, v)
        print('---' * 30)
    print(f'total: {i}')
