import logging as _logging

from npuperf.classes.stages import *
from npuperf.utils import pickle_deepcopy

################### initializing the log ###################
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(
    level=_logging_level,
    format=_logging_format,
)
logger = _logging.getLogger(__name__)

################### initializing acc path and json NN dir path ###################
json_path = 'npuperf/inputs/hhb_networks/resnet50.json'
accelerator_dir_path = 'npuperf.inputs.HW.?'
wl_name = json_path.split('/')[-1][:-5]

################### initializing json NN list and stage pipeline ###################
HW = [
    'example_wioGB',
    'Ascend_like',
    'Edge_TPU_like',
    'TPU_like',
    'Meta_prototype',
]
list_of_callables = [
    AcceleratorParserStage,
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

################### initializing mainstage pattern ###################
mainstage = MainStage(
    list_of_callables=list_of_callables,
    loma_lpf_limit=6,
    json_NN = json_path,
)

################### for loop iterating workloads and START RUNNING ###################
for id, hardware in enumerate(HW, start=1):
    this_stage = pickle_deepcopy(mainstage)
    accelerator_path = accelerator_dir_path.replace('?', f'{hardware}')
    hw_name = accelerator_path.split('.')[-1]
    experiment_id = f"iter_HW-{wl_name}/{hw_name}--{wl_name}"
    pkl_name = 'saved_list_of_cmes'
    this_stage.kwargs['dump_filename_pattern'] = f"outputs/{experiment_id}/?.json"
    this_stage.kwargs['pickle_filename'] = f"outputs/{experiment_id}/{pkl_name}.pickle"
    this_stage.kwargs['plot_filename_pattern'] = f'result_plot/{experiment_id}/?.png'
    this_stage.kwargs['accelerator_path'] = accelerator_path
    logger.info('###' * 15 + f'\tRunning experiment ({id} / {len(HW)}): {experiment_id} ...\t' + '###' * 15)
    this_stage.run()
