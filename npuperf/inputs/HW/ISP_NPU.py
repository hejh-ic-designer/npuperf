import os
from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from npuperf.classes.hardware.architecture.memory_level import MemoryLevel
from npuperf.classes.hardware.architecture.operational_unit import Multiplier
from npuperf.classes.hardware.architecture.operational_array import MultiplierArray
from npuperf.classes.hardware.architecture.memory_instance import MemoryInstance
from npuperf.classes.hardware.architecture.accelerator import Accelerator
from npuperf.classes.hardware.architecture.core import Core


def memory_hierarchy_dut(multiplier_array):
    """Memory hierarchy variables"""
    ''' size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) '''
    reg_data_shceduler = MemoryInstance(name="rf_data_shceduler", size=2 * 1024 * 8, r_bw=128 * 16 * 8, w_bw=128 * 16 * 8,
                                        r_cost=1.28, w_cost=1.3, area=0, bank=1, random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)

    reg_O1 = MemoryInstance(name="rf_1B", size=16, r_bw=16, w_bw=16, r_cost=0.01, w_cost=0.01, area=0, bank=1,
                            random_bank_access=False, r_port=2, w_port=2, rw_port=0, latency=1)

    ##################################### on-chip memory hierarchy building blocks #####################################

    # sram_OR = \
    #     MemoryInstance(name="sram_256KB_OR", size=256 * 1024 * 8, r_bw=128 * 16 * 8, w_bw=128 * 16 * 8, r_cost=26.01 * 16, w_cost=23.65 * 16, area=0, bank=1,
    #                    random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1, min_r_granularity=64, min_w_granularity=64)

    sram_W = \
        MemoryInstance(name="sram_64KB_W", size=64 * 1024 * 8, r_bw=16 * 8 * 2, w_bw=16 * 8 * 2, r_cost=6.27 * 8, w_cost=13.5 * 8, area=0, bank=1,
                       random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1, min_r_granularity=64, min_w_granularity=64)

    sram_IO = \
        MemoryInstance(name="sram_1MB_A", size=1* 1024* 1024 * 8, r_bw=128 * 16 * 8, w_bw=128 * 16 * 8, r_cost=15.4 * 8, w_cost=26.6 * 8, area=0, bank=1,
                       random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1, min_r_granularity=64, min_w_granularity=64)


    #######################################################################################################################

    dram = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=700, w_cost=750, area=0, bank=1,
                          random_bank_access=False, r_port=0, w_port=0, rw_port=1, latency=1)

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    '''
    fh: from high = wr_in_by_high
    fl: from low = wr_in_by_low
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    '''
    memory_hierarchy_graph.add_memory(memory_instance=reg_O1, operands=('O',),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
                                      served_dimensions={(0, 1, 0, 0)})
    memory_hierarchy_graph.add_memory(memory_instance=reg_data_shceduler, operands=('I1',),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                      served_dimensions='all')

    ##################################### on-chip highest memory hierarchy initialization #####################################

    # memory_hierarchy_graph.add_memory(memory_instance=sram_OR, operands=('R',),
    #                                   port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
    #                                   served_dimensions='all')

    memory_hierarchy_graph.add_memory(memory_instance=sram_W, operands=('I2',),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                      served_dimensions='all')

    memory_hierarchy_graph.add_memory(memory_instance=sram_IO, operands=('I1', 'O'),
                                      port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
                                                  {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
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
    dimensions = {'D1': 4, 'D2': 4, 'D3': 32, 'D4': 4}      # K, C, OX, OY

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
