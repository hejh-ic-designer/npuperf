import logging as _logging

from npuperf.classes.stages import *

# by gyp
# 这里是log的设置，可以设置log的级别，格式等。下面这两行设置log的级别为INFO，只会打印大于等于INFO级别的log
# 设置log的格式为：时间 - 模块名.函数名 +行号 - log级别 - log信息
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(
    level=_logging_level,
    format=_logging_format,
)
# 创建了一个logger，名字为__name__，这个名字是当前模块的名字，也就是tools.main_LBL
logger = _logging.getLogger(__name__)

#! here to setting DLA path
accelerator_path='npuperf.inputs.HW.Meta_prototype_3ddram'
hw_name = accelerator_path.split('.')[-1]

#! here to setting WL path
workload_path = f'npuperf.inputs.WL_fromjson.{hw_name}.workload_mv1'
# workload_path = f'npuperf.inputs.WL_fromjson.{hw_name}.workload_fsrcnn2x'
# workload_path = f'npuperf.inputs.WL_fromjson.{hw_name}.workload_fcn8s'
# workload_path = f'npuperf.inputs.WL_fromjson.{hw_name}.workload_resnet50'
# workload_path = f'npuperf.inputs.WL_fromjson.{hw_name}.workload_conv_test'

wl_name = "_".join(workload_path.split('.')[-1].split('_')[1:])
experiment_id = f"{hw_name}--{wl_name}"

mainstage = MainStage(
    [
        AcceleratorParserStage,
        WorkloadParserStage,
        SumAndSaveAllLayersStage,
        PickleSaveStage,
        # SimpleSaveStage,
        CompleteSaveStage,
        PlotTemporalMappingsStage,
        WorkloadStage,
        SpatialMappingConversionStage,
        MinimalLatencyStage,
        LomaStage,
        ZigZagCostModelStage
    ],
    loma_lpf_limit=6,  # required by LOMA stage, set to 6 - 8 will be nice.
    workload_path=workload_path,
    accelerator_path=accelerator_path,
    dump_filename_pattern=f"outputs/{experiment_id}/?.json",
    pickle_filename=f"outputs/{experiment_id}/{experiment_id}.pickle",
    plot_filename_pattern=f'result_plot/{experiment_id}/?.png',
    # execution_layers = [8],
)

if __name__ == '__main__':
    logger.info(f'Runing experiment: {experiment_id} ...')
    mainstage.run()
