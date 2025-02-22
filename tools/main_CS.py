# Case Studies
import logging as _logging
from npuperf.utils import pickle_deepcopy
from npuperf.classes.stages import *
from tools.Mem_hier_gen import CS0, CS1, CS2, CS3, CS4, CS5, CS6, CS7, nb_of_cases
#CS3, CS2, CS1
################### initializing the log ###################
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

################### initializing NN dir path ###################
# workload_path = 'workload_tst.case_study.workload_fsrcnn2x'
workload_path = 'workload_tst.case_study.workload_mv1'
wl_name = "_".join(workload_path.split('.')[-1].split('_')[1:])

################### initializing stage pipeline ###################

list_of_callables = [
    HardwareGeneratorStage, MappingGeneratorStage, WorkloadParserStage, SumAndSaveAllLayersStage, WorkloadStage, SpatialMappingConversionStage,
    MinimalLatencyStage, LomaStage, ZigZagCostModelStage
]

################### initializing mainstage pattern ###################
mainstage = MainStage(
    list_of_callables=list_of_callables,
    loma_lpf_limit=6,
    workload_path=workload_path,
)

################### for loop iterating HW_gens and START RUNNING ###################
#! here to change CASE !!!
CASE_list = [
    # CS0,
    # CS1,
    # CS2,
    # CS3,
    # CS4,
    # CS5,
    # CS6,
    CS7,
]

for CASE in CASE_list:

    for id, hardware in enumerate(CASE().gen_cases(), start=1):
        this_stage = pickle_deepcopy(mainstage)
        hw_name = hardware['name']
        experiment_id = f"Case_Study_{str(CASE.__name__)[-1]}--{wl_name}"
        this_stage.kwargs['dump_filename_pattern'] = f"outputs/{experiment_id}/{hw_name}_?.json"
        this_stage.kwargs['json_HW'] = hardware
        logger.info('###' * 15 + f'\tRunning experiment ({id} / {nb_of_cases[str(CASE.__name__)]}): {experiment_id} ...\t' + '###' * 15)
        this_stage.run()
    logger.info(f' ---------------{CASE}  DONE  --------------- ')
