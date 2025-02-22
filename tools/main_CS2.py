# Case Study 2
import logging as _logging
from npuperf.utils import pickle_deepcopy
from npuperf.classes.stages import *
from tools.Mem_hier_gen_cs2 import CASE_gen

################### initializing the log ###################
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

################### initializing NN dir path ###################
# workload_path = 'npuperf.inputs.WL_fromjson.case_study.workload_fsrcnn2x'
workload_path = 'npuperf.inputs.WL_fromjson.cs2.workload_test_matmul'
wl_name = "_".join(workload_path.split('.')[-1].split('_')[1:])

################### initializing stage pipeline ###################

list_of_callables = [
    HardwareGeneratorStage, 
    MappingGeneratorStage, 
    WorkloadParserAddSMStage, 
    SumAndSaveAllLayersStage, 
    CompleteSaveStage,
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
    loma_lpf_limit=8,
    workload_path=workload_path,
)

################### for loop iterating HW_gens and START RUNNING ###################


for id, hardware in enumerate(CASE_gen(ratio=0.5).gen_cases(), start=1):    #! here to change ratio
    this_stage = pickle_deepcopy(mainstage)
    hw_name = hardware['name']
    experiment_id = f"{hw_name}--{wl_name}"
    this_stage.kwargs['dump_filename_pattern'] = f"outputs/CS2/{experiment_id}/?.json"
    this_stage.kwargs['plot_filename_pattern'] = f'result_plot/CS2/{experiment_id}/?.png'
    this_stage.kwargs['json_HW'] = hardware
    logger.info('###' * 15 + f'\tRunning experiment {id}: {experiment_id} ...\t' + '###' * 15)
    this_stage.run()
logger.info(f' ---------------  DONE  --------------- ')
