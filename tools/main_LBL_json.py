import logging as _logging

from npuperf.classes.stages import *

_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(
    level=_logging_level,
    format=_logging_format,
)
logger = _logging.getLogger(__name__)

# accelerator_path='inputs.HW.Meta_prototype'
accelerator_path = 'npuperf.inputs.HW.example_wioGB'
json_path = 'npuperf/inputs/hhb_networks/fcn8s.json'

hw_name = accelerator_path.split('.')[-1]
wl_name = json_path.split('/')[-1][:-5]
experiment_id = f"{hw_name}--{wl_name}"
pkl_name = f'{experiment_id}-saved_list_of_cmes'

mainstage = MainStage(
    [
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
    ],
    loma_lpf_limit=6,  # required by LOMA stage, set to 6 - 8 will be nice.
    json_NN=json_path,
    accelerator_path=accelerator_path,
    dump_filename_pattern=f"outputs/{experiment_id}/?.json",
    pickle_filename=f"outputs/{experiment_id}/{pkl_name}.pickle",
    plot_filename_pattern=f'result_plot/{experiment_id}/?.png',
)

if __name__ == '__main__':
    logger.info(f'Runing experiment: {experiment_id} ...')
    mainstage.run()
