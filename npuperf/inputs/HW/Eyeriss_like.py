import os
from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from npuperf.classes.hardware.architecture.memory_level import MemoryLevel
from npuperf.classes.hardware.architecture.operational_unit import Multiplier
from npuperf.classes.hardware.architecture.operational_array import MultiplierArray
from npuperf.classes.hardware.architecture.memory_instance import MemoryInstance
from npuperf.classes.hardware.architecture.accelerator import Accelerator
from npuperf.classes.hardware.architecture.core import Core


def memory_hierarchy_latency_test1(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """
    # I reg & W reg
    rf1 = MemoryInstance(
        name="rf_64B", size=512, r_bw=8, w_bw=8, r_cost=1.0, w_cost=1.5, area=0.3,
        r_port=1, w_port=1, rw_port=0, latency=1, bank=1, random_bank_access=False)  # rd E per bit 0.125

    # O reg
    rf2 = MemoryInstance(
        name="rf_16B", size=128, r_bw=24, w_bw=24, r_cost=1.5, w_cost=2, area=0.95,
        r_port=1, w_port=1, rw_port=1, latency=1, bank=1, random_bank_access=False)  # rd E per bit 0.0625

    # lb1 = MemoryInstance(name="sram_64KB", size=524288, r_bw=128, w_bw=128, r_cost=20, w_cost=25, area=6, r_port=1, w_port=1, rw_port=0, latency=1)  # rd E per bit 0.16
    # O LB
    lb2 = MemoryInstance(
        name="sram_8KB", size=65536, r_bw=128, w_bw=128, r_cost=10, w_cost=15, area=3,
        r_port=0, w_port=0, rw_port=2, latency=1, bank=1, random_bank_access=True)  # rd E per bit 0.08

    # W LB
    lb2_64KB = MemoryInstance(
        name="sram_64KB", size=524288, r_bw=128, w_bw=128, r_cost=20, w_cost=25, area=6,
        r_port=1, w_port=1, rw_port=0, latency=1, bank=1, random_bank_access=True)  # rd E per bit 0.08

    gb = MemoryInstance(
        name="sram_1M_A", size=8388608, r_bw=384, w_bw=384, r_cost=100, w_cost=130, area=25,
        r_port=0, w_port=0, rw_port=2, latency=1, bank=1, random_bank_access=True)  # rd E per bit 0.26

    dram = MemoryInstance(
        name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=1000, w_cost=1000, area=0,
        r_port=0, w_port=0, rw_port=1, latency=1, bank=1, random_bank_access=False)  # rd E per bit 16

    memory_hierarchy_graph = MemoryHierarchy(
        operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high
    fl: from low = wr_in_by_low
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        # I reg 64B
        memory_instance=rf1,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1",
                    "fl": None, "th": None},),
        served_dimensions=set(),
    )
    memory_hierarchy_graph.add_memory(
        # W reg 64B
        memory_instance=rf1,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1",
                    "fl": None, "th": None},),
        served_dimensions=set(),
    )
    memory_hierarchy_graph.add_memory(
        # O reg 16B
        memory_instance=rf2,
        operands=("O",),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "r_port_1",
                "fl": "w_port_1", "th": "rw_port_1"},
        ),
        served_dimensions=set(),
    )

    memory_hierarchy_graph.add_memory(
        # O LB 8KB
        memory_instance=lb2,
        operands=("O",),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "rw_port_2",
                "fl": "rw_port_2", "th": "rw_port_1", },
        ),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        # W LB 64KB
        memory_instance=lb2_64KB,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1",
                    "fl": None, "th": None},),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        # IO GB 1M
        memory_instance=gb,
        operands=("I1", "O"),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "rw_port_2", "fl": None, "th": None},
            {"fh": "rw_port_1", "tl": "rw_port_2",
                "fl": "rw_port_2", "th": "rw_port_1", },
        ),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        # DRAM
        memory_instance=dram,
        operands=("I1", "I2", "O"),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
            {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
            {"fh": "rw_port_1", "tl": "rw_port_1",
                "fl": "rw_port_1", "th": "rw_port_1", },
        ),
        served_dimensions="all",
    )

    return memory_hierarchy_graph


def multiplier_array_latency_test1():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {"D1": 14, "D2": 12}  # ??? {'K': }
    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def cores_dut():
    multiplier_array1 = multiplier_array_latency_test1()
    memory_hierarchy1 = memory_hierarchy_latency_test1(multiplier_array1)
    core1 = Core(1, multiplier_array1, memory_hierarchy1)
    return {core1}


cores = cores_dut()
acc_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator(acc_name, cores, None)
