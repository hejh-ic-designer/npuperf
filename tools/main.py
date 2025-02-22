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
    HardwareModifierStage,
    MappingGeneratorStage,
    JsonWorkloadParserStage,
    WorkloadParserStage,
    SumAndSaveAllLayersStage,
    PickleSaveStage,
    CompleteSaveStage,
    # SimpleSaveStage,
    PlotTemporalMappingsStage,
    WorkloadStage,
    SpatialMappingConversionStage,
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
