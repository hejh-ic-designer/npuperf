drams = [
    {
        'op': "W/I/O",
        'bandwidth': 96
    },
]

gbs = [[0.5, 128], [2, 256], [4, 512]]

lbs = [16, 64, 256]

MAC_unroll = {"K": 8, "C": 32, "OX": 2, "OY": 2}

nb_of_cases = {
    'CS0': 81,
    'CS1': 27,
    'CS2': 81,
    'CS3': 27,
    'CS4': 27,
    'CS5': 9,
    'CS6': 81,
    'CS7': 27,
}
nb_of_mems = {
    'CS0': ['L2_WIO', 'W', 'I', 'O'],
    'CS1': ['L2_WIO', 'W', 'IO'],
    'CS2': ['L2_IO', 'W', 'I', 'O'],
    'CS3': ['L2_IO', 'W', 'IO'],
    'CS4': ['W', 'I', 'O'],
    'CS5': ['W', 'IO'],
    'CS6': ['L2_W', 'W', 'I', 'O'],
    'CS7': ['L2_W', 'W', 'IO'],
}


class CS0:

    def __init__(self) -> None:
        self.hw_topology = {
            "MAC_unroll": MAC_unroll,
            "local_buffers": [{
                "op": "W",
            }, {
                "op": "I",
            }, {
                "op": "O",
            }],
            "global_buffer": {
                "op": "O/W/I",
            },
            "dram": drams[0]
        }

    def gen_cases(self):
        for gb in gbs:
            for w_lb in lbs:
                for i_lb in lbs:
                    for o_lb in lbs:
                        new_hw = self.hw_topology.copy()
                        new_hw["name"] = f'WIO{gb[0]}+W{w_lb}_I{i_lb}_O{o_lb}'
                        new_hw['local_buffers'][0]["size"] = w_lb
                        new_hw['local_buffers'][1]["size"] = i_lb
                        new_hw['local_buffers'][2]["size"] = o_lb
                        new_hw['global_buffer']["size"] = gb[0]
                        new_hw['global_buffer']["bandwidth"] = gb[1]
                        yield new_hw


class CS1:

    def __init__(self) -> None:
        self.hw_topology = {
            "MAC_unroll": MAC_unroll,
            "local_buffers": [{
                "op": "W",
            }, {
                "op": "I/O",
            }],
            "global_buffer": {
                "op": "O/W/I",
            },
            "dram": drams[0]
        }

    def gen_cases(self):
        for gb in gbs:
            for w_lb in lbs:
                for io_lb in lbs:
                    new_hw = self.hw_topology.copy()
                    new_hw["name"] = f'WIO{gb[0]}+W{w_lb}_IO{io_lb}'
                    new_hw['local_buffers'][0]["size"] = w_lb
                    new_hw['local_buffers'][1]["size"] = io_lb
                    new_hw['global_buffer']["size"] = gb[0]
                    new_hw['global_buffer']["bandwidth"] = gb[1]
                    yield new_hw


class CS2:

    def __init__(self) -> None:
        self.hw_topology = {
            "MAC_unroll": MAC_unroll,
            "local_buffers": [{
                "op": "W",
            }, {
                "op": "I",
            }, {
                "op": "O",
            }],
            "global_buffer": {
                "op": "I/O",
            },
            "dram": drams[0]
        }

    def gen_cases(self):
        for gb in gbs:
            for w_lb in lbs:
                for i_lb in lbs:
                    for o_lb in lbs:
                        new_hw = self.hw_topology.copy()
                        new_hw["name"] = f'IO{gb[0]}+W{w_lb}_I{i_lb}_O{o_lb}'
                        new_hw['local_buffers'][0]["size"] = w_lb
                        new_hw['local_buffers'][1]["size"] = i_lb
                        new_hw['local_buffers'][2]["size"] = o_lb
                        new_hw['global_buffer']["size"] = gb[0]
                        new_hw['global_buffer']["bandwidth"] = gb[1]
                        yield new_hw


class CS3:

    def __init__(self) -> None:
        self.hw_topology = {
            "MAC_unroll": MAC_unroll,
            "local_buffers": [{
                "op": "W",
            }, {
                "op": "I/O",
            }],
            "global_buffer": {
                "op": "I/O",
            },
            "dram": drams[0]
        }

    def gen_cases(self):
        for gb in gbs:
            for w_lb in lbs:
                for io_lb in lbs:
                    new_hw = self.hw_topology.copy()
                    new_hw["name"] = f'IO{gb[0]}+W{w_lb}_IO{io_lb}'
                    new_hw['local_buffers'][0]["size"] = w_lb
                    new_hw['local_buffers'][1]["size"] = io_lb
                    new_hw['global_buffer']["size"] = gb[0]
                    new_hw['global_buffer']["bandwidth"] = gb[1]
                    yield new_hw


class CS4:

    def __init__(self) -> None:
        self.hw_topology = {
            "MAC_unroll": MAC_unroll,
            "local_buffers": [{
                "op": "W",
            }, {
                "op": "I",
            }, {
                "op": "O",
            }],
            "dram": drams[0]
        }

    def gen_cases(self):
        for w_lb in lbs:
            for i_lb in lbs:
                for o_lb in lbs:
                    new_hw = self.hw_topology.copy()
                    new_hw["name"] = f'W{w_lb}_I{i_lb}_O{o_lb}'
                    new_hw['local_buffers'][0]["size"] = w_lb
                    new_hw['local_buffers'][1]["size"] = i_lb
                    new_hw['local_buffers'][2]["size"] = o_lb
                    yield new_hw


class CS5:

    def __init__(self) -> None:
        self.hw_topology = {
            "MAC_unroll": MAC_unroll,
            "local_buffers": [{
                "op": "W",
            }, {
                "op": "I/O",
            }],
            "dram": drams[0]
        }

    def gen_cases(self):
        for w_lb in lbs:
            for io_lb in lbs:
                new_hw = self.hw_topology.copy()
                new_hw["name"] = f'W{w_lb}_IO{io_lb}'
                new_hw['local_buffers'][0]["size"] = w_lb
                new_hw['local_buffers'][1]["size"] = io_lb
                yield new_hw

class CS6:

    def __init__(self) -> None:
        self.hw_topology = {
            "MAC_unroll": MAC_unroll,
            "local_buffers": [{
                "op": "W",
            }, {
                "op": "I",
            }, {
                "op": "O",
            }],
            "global_buffer": {
                "op": "W",
            },
            "dram": drams[0]
        }

    def gen_cases(self):
        for gb in gbs:
            for w_lb in lbs:
                for i_lb in lbs:
                    for o_lb in lbs:
                        new_hw = self.hw_topology.copy()
                        new_hw["name"] = f'W{gb[0]}+W{w_lb}_I{i_lb}_O{o_lb}'
                        new_hw['local_buffers'][0]["size"] = w_lb
                        new_hw['local_buffers'][1]["size"] = i_lb
                        new_hw['local_buffers'][2]["size"] = o_lb
                        new_hw['global_buffer']["size"] = gb[0]
                        new_hw['global_buffer']["bandwidth"] = gb[1]
                        yield new_hw


class CS7:

    def __init__(self) -> None:
        self.hw_topology = {
            "MAC_unroll": MAC_unroll,
            "local_buffers": [{
                "op": "W",
            }, {
                "op": "I/O",
            }],
            "global_buffer": {
                "op": "W",
            },
            "dram": drams[0]
        }

    def gen_cases(self):
        for gb in gbs:
            for w_lb in lbs:
                for io_lb in lbs:
                    new_hw = self.hw_topology.copy()
                    new_hw["name"] = f'W{gb[0]}+W{w_lb}_IO{io_lb}'
                    new_hw['local_buffers'][0]["size"] = w_lb
                    new_hw['local_buffers'][1]["size"] = io_lb
                    new_hw['global_buffer']["size"] = gb[0]
                    new_hw['global_buffer']["bandwidth"] = gb[1]
                    yield new_hw



if __name__ == '__main__':
    i = 0
    for MEM_hier in CS7().gen_cases():
        i += 1
        print(MEM_hier)
    print(f'total: {i}')
