import argparse
import logging as _logging

from npuperf.classes.stages import *

_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Setup npuperf inputs")
parser.add_argument('--nn', metavar='Network name', required=True, help='module name to hhb networks, e.g. fsrcnn2x')
parser.add_argument('--hw', metavar='Hardware name', required=True, help='module name to the accelerator, e.g. example_wioGB')
parser.add_argument(
    '--flow',
    metavar='Temporal data flow',
    choices=['O', 'W'],
    required=False,
    help=
    'three types of Temporal mapping search method, default to Free completely LOMA engine, set "O" to Fully output stationary, set "W" to Fully Weight stationary'
)
parser.add_argument('--gb_size',
                    metavar='hardware config info: the global buffer size (MB)',
                    required=False,
                    help='Optional: change Global Buffer size based on selected hardware, e.g. 3')
parser.add_argument('--gb_bw',
                    metavar='hardware config info: the global buffer bandwidth (bit/cycle)',
                    required=False,
                    help='Optional: change Global Buffer bandwidth based on selected hardware, e.g. 256')
parser.add_argument('--dram_bw',
                    metavar='hardware config info: the dram bandwidth (bit/cycle)',
                    required=False,
                    help='Optional: change dram bandwidth based on selected hardware, e.g. 256')

args = parser.parse_args()
experiment_id = f"{args.hw}--{args.nn}--{args.flow}" if args.flow else f"{args.hw}--{args.nn}"

StagesPipeline = [
    HardwareModifierStage, # 解析硬件py
    MappingGeneratorStage, # 解析各个算子的mapping信息，每个算子在硬件上的mapping方式都是固定死的
    JsonWorkloadParserStage, # 解析工作负载json文件，生成NPUPerf能识别的对象，其实这里相当于从外部到NPUPerf内部的转换接口
    WorkloadParserStage, # 解析工作负载，生成NPUPerfWorkload对象，包括pr ir r的处理都在这里
    SumAndSaveAllLayersStage, #这个负责存整个计算结果（总体的），按理说应该有很多个list组成，但实际上好像只有一个all_layer，应该是在下游stage做了处理(MinimalLatencyStage只挑了一个最小延时的结果) 
    PickleSaveStage, # 将结果pickle化保存，我没看出来和上一个stage保存内容的区别
    CompleteSaveStage, #还是做保存，不确定是否保存单层
    # SimpleSaveStage,
    PlotTemporalMappingsStage,
    WorkloadStage, #可以只执行其中的几层，会对两个输入操作数的硬件和workload做处理，修改加速器内存结构,直接删除I2，复制一份I1作为I2
    SpatialMappingConversionStage, # 生成一个spatial mapping对象，包含了每个算子在硬件上的空间映射信息,在对象初始化时，已经把OX和OY的pr部分进行解耦
    MinimalLatencyStage,
    LomaStage,
    ZigZagCostModelStage
]
if args.flow:
    StagesPipeline[-2] = StationaryLomaStage

mainstage = MainStage(
    list_of_callables=StagesPipeline,
    export_wl=False,  # export wl in JsonWorkloadParserStage
    args=args,
    loma_lpf_limit=6,  # required by LOMA stage, set to 6 - 8 will be nice.
    stationary=args.flow,
    json_NN=f'npuperf/inputs/hhb_networks/{args.nn}.json',
    dump_filename_pattern=f"outputs/{experiment_id}/?.json",
    pickle_filename=f"outputs/{experiment_id}/{experiment_id}.pickle",
    plot_filename_pattern=f'result_plot/{experiment_id}/?.png',
    # execution_layers=[1, 2, 3]  # 执行的部分 layers，若为None则执行All layers
)

logger.info(f'Runing experiment: {experiment_id} ...')
mainstage.run()
