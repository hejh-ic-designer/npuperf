import argparse
import logging as _logging

from npuperf.classes.stages import *

_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Setup npuperf inputs")
parser.add_argument('--nn', metavar='Network name', required=True, help='module name to hhb networks, e.g. fsrcnn2x')
parser.add_argument('--hw',
                    metavar='Hardware config file name',
                    required=True,
                    help='file name to the user-defined json accelerator config file, e.g. example_wioGB')
parser.add_argument(
    '--flow',
    metavar='Temporal data flow',
    choices=['O', 'W'],
    required=False,
    help=
    'three types of Temporal mapping search method, default to Free completely LOMA engine, set "O" to Fully output stationary, set "W" to Fully Weight stationary'
)

args = parser.parse_args()
experiment_id = f"{args.hw}_gen--{args.nn}--{args.flow}" if args.flow else f"{args.hw}_gen--{args.nn}"

StagesPipeline = [
    HardwareGeneratorStage,
    MappingGeneratorStage,
    JsonWorkloadParserStage,
    WorkloadParserStage,
    SumAndSaveAllLayersStage,
    PickleSaveStage,
    # CompleteSaveStage,
    SimpleSaveStage,
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
    # export_wl = True,   # export wl in JsonWorkloadParserStage
    loma_lpf_limit=6,  # required by LOMA stage, set to 6 - 8 will be nice.
    stationary=args.flow,
    json_NN=f'npuperf/inputs/hhb_networks/{args.nn}.json',
    json_HW=f'npuperf/inputs/hw_config/{args.hw}.json',
    dump_filename_pattern=f"outputs/{experiment_id}/?.json",
    pickle_filename=f"outputs/{experiment_id}/{experiment_id}.pickle",
    plot_filename_pattern=f'result_plot/{experiment_id}/?.png',
    # execution_layers=[1, 2, 3]  # 执行的部分 layers，若为None则执行All layers
)

logger.info(f'Runing experiment: {experiment_id} ...')
mainstage.run()
