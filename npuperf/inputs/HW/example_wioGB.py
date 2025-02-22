import os
from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from npuperf.classes.hardware.architecture.operational_unit import Multiplier
from npuperf.classes.hardware.architecture.operational_array import MultiplierArray
from npuperf.classes.hardware.architecture.memory_instance import MemoryInstance
from npuperf.classes.hardware.architecture.accelerator import Accelerator
from npuperf.classes.hardware.architecture.core import Core


def memory_hierarchy_dut(multiplier_array) -> MemoryHierarchy:
    """
    设置内存实例, 有reg, sram, DRAM三个级别, 然后把MemoryInstance 连接成 Memory hierarchy graph \\
    Memory hierarchy variables  \\
    size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy)
    """

    reg_I1 = MemoryInstance(name="rf_1B", size=8, r_bw=8, w_bw=8, r_cost=0.01, w_cost=0.01, area=0, bank=1,
                             random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)

    reg_W1 = MemoryInstance(name="rf_1B", size=8, r_bw=8, w_bw=8, r_cost=0.01, w_cost=0.01, area=0, bank=1,
                             random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)

    reg_O1 = MemoryInstance(name="rf_4B", size=32, r_bw=32, w_bw=32, r_cost=0.03, w_cost=0.03, area=0, bank=1,
                            random_bank_access=False, r_port=2, w_port=2, rw_port=0, latency=1)

    ##################################### on-chip memory hierarchy building blocks #####################################

    # input buffer 64KB
    sram_64KB_1K_1r_1w_I = \
        MemoryInstance(name="sram_64KB", size=64 * 1024 * 8, r_bw=128 * 8, w_bw=128 * 8, r_cost=45.3, w_cost=45.3*1.5, area=0, bank=1,
                       random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1, min_r_granularity=64, min_w_granularity=64)

    # weight buffer 64KB
    sram_64KB_2K_1r_1w_W = \
        MemoryInstance(name="sram_64KB", size=64 * 1024 * 8, r_bw=256 * 8, w_bw=256 * 8, r_cost=64, w_cost=64*1.5, area=0, bank=1,
                       random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1, min_r_granularity=64, min_w_granularity=64)

    # acc buffer 128KB
    sram_128KB_1K_1r_1w_O = \
        MemoryInstance(name="sram_128KB", size=128 * 1024 * 8, r_bw=32 * 32, w_bw=32 * 32, r_cost=64, w_cost=64*1.5, area=0, bank=1,
                       random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1, min_r_granularity=64, min_w_granularity=64)

    # Global Buffer 1MB bw: 128b/c
    sram_1M_128_1r_1w_A = \
        MemoryInstance(name="sram_1MB_A", size=1024 * 1024 * 8, r_bw=128, w_bw=128, r_cost=64, w_cost=64*1.5, area=0, bank=1,
                       random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1, min_r_granularity=64, min_w_granularity=64)

    #######################################################################################################################

    # DRAM 4GB bw: 96b/c
    dram = MemoryInstance(name="dram", size=1024 * 1024 * 1024 * 4 * 8, r_bw=96, w_bw=96, r_cost=700, w_cost=750, area=0, bank=1,
                          random_bank_access=False, r_port=0, w_port=0, rw_port=1, latency=1)

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    '''
    fh: from high = wr_in_by_high
    fl: from low = wr_in_by_low
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    '''
    # memory_hierarchy_graph.add_memory(memory_instance=reg_I1, operands=('I1',),
    #                                   port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
    #                                   served_dimensions={(1, 0, 0, 0)})

    memory_hierarchy_graph.add_memory(memory_instance=reg_W1, operands=('I2',),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                      served_dimensions={(0, 0, 1, 0), (0, 0, 0, 1)})
    memory_hierarchy_graph.add_memory(memory_instance=reg_O1, operands=('O',),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
                                      served_dimensions={(0, 1, 0, 0)})

    ##################################### on-chip highest memory hierarchy initialization #####################################

    memory_hierarchy_graph.add_memory(memory_instance=sram_64KB_2K_1r_1w_W, operands=('I2',),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                      served_dimensions='all')

    memory_hierarchy_graph.add_memory(memory_instance=sram_64KB_1K_1r_1w_I, operands=('I1',),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                      served_dimensions='all')

    memory_hierarchy_graph.add_memory(memory_instance=sram_128KB_1K_1r_1w_O, operands=('O',),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                                      served_dimensions='all')

    ####################################################################################################################

    memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'),
                                      port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                  {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                  {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),
                                      served_dimensions='all')

    return memory_hierarchy_graph


def multiplier_array_dut():
    """ Multiplier array variables """
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.04
    multiplier_area = 1

    dimensions = {'D1': 8, 'D2': 32, 'D3': 2, 'D4': 2}  # {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)}

    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def cores():
    multiplier_array1 = multiplier_array_dut()
    memory_hierarchy1 = memory_hierarchy_dut(multiplier_array1)
    core1 = Core(1, multiplier_array1, memory_hierarchy1)
    return {core1}


cores = cores()
global_buffer = None
acc_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator(acc_name, cores, global_buffer)

a = 1
